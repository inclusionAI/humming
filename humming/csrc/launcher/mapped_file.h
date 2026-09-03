#pragma once

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

class MappedFile {
public:
  explicit MappedFile(const std::string &path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw_error("open", path, errno);

    struct stat st;
    if (fstat(fd, &st) != 0) {
      int error = errno;
      close(fd);
      throw_error("fstat", path, error);
    }
    if (st.st_size <= 0) {
      close(fd);
      throw std::runtime_error("empty file: " + path);
    }

    size_ = static_cast<size_t>(st.st_size);
    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    int error = errno;
    close(fd);
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      throw_error("mmap", path, error);
    }
  }

  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;

  ~MappedFile() {
    if (data_ != nullptr) munmap(data_, size_);
  }

  const void *data() const { return data_; }
  size_t size() const { return size_; }

private:
  [[noreturn]] static void throw_error(const char *operation, const std::string &path, int error) {
    throw std::runtime_error(
        std::string(operation) + " failed for " + path + ": " + std::strerror(error));
  }

  void *data_ = nullptr;
  size_t size_ = 0;
};
