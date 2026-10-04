"""CPU limits shared by JIT compilation and the CI startup report."""

import os
import re
from pathlib import Path


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _decode_mount_path(value: str) -> Path:
    return Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value))


def get_cgroup_cpu_count() -> int | None:
    memberships = {}
    for line in _read_text(Path("/proc/self/cgroup")).splitlines():
        _, controllers, path = line.split(":", 2)
        if not controllers or "cpu" in controllers.split(","):
            memberships["cgroup2" if not controllers else "cgroup"] = Path(path)

    limits = []
    for line in _read_text(Path("/proc/self/mountinfo")).splitlines():
        mount, separator, filesystem = line.partition(" - ")
        if not separator:
            continue
        fields = filesystem.split()
        filesystem_type = fields[0]
        if filesystem_type not in memberships:
            continue
        if filesystem_type == "cgroup" and "cpu" not in fields[2].split(","):
            continue
        mount_fields = mount.split()
        root = _decode_mount_path(mount_fields[3])
        mount_point = _decode_mount_path(mount_fields[4])
        membership = memberships[filesystem_type]
        try:
            relative_path = membership.relative_to(root)
        except ValueError:
            # A cgroup namespace can expose membership relative to its own root.
            relative_path = membership.relative_to("/")
        if ".." in relative_path.parts:
            relative_path = Path(".")
        directory = mount_point / relative_path
        while True:
            try:
                if filesystem_type == "cgroup2":
                    quota, period = _read_text(directory / "cpu.max").split()
                else:
                    quota = _read_text(directory / "cpu.cfs_quota_us")
                    period = _read_text(directory / "cpu.cfs_period_us")
                if int(quota) > 0 and int(period) > 0:
                    limits.append(max(1, int(quota) // int(period)))
            except ValueError:
                pass  # Missing or unlimited quota.
            if directory == mount_point:
                break
            directory = directory.parent
    return min(limits) if limits else None


def get_cpu_affinity_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def get_available_cpu_count() -> int:
    available = min(os.cpu_count() or 1, get_cpu_affinity_count())
    quota = get_cgroup_cpu_count()
    return max(1, min(available, quota)) if quota is not None else max(1, available)


def get_parallel_build_workers() -> int:
    available = get_available_cpu_count()
    if os.environ.get("HUMMING_DISABLE_PARALLEL_BUILD", "0") == "1":
        return 1
    value = os.environ.get("HUMMING_MAX_PARALLEL_BUILD_WORKERS")
    if value is None:
        return available
    try:
        requested = int(value)
        if requested > 0:
            return min(available, requested)
    except ValueError:
        pass
    raise ValueError("HUMMING_MAX_PARALLEL_BUILD_WORKERS must be a positive integer")


if __name__ == "__main__":
    print(f"System CPU threads: {os.cpu_count() or 1}")
    print(f"CPU affinity threads: {get_cpu_affinity_count()}")
    print(f"Cgroup CPU quota threads (rounded down, minimum 1): {get_cgroup_cpu_count() or 'unlimited'}")
    print(f"Available CPU threads: {get_available_cpu_count()}")
    print(f"Parallel compilation worker limit: {get_parallel_build_workers()}")
