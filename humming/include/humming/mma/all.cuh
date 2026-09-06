#pragma once

#include <humming/mma/mxmma.cuh>
#include <humming/mma/mxumma.cuh>
#include <humming/mma/wgmma.cuh>
#include <humming/mma/wmma.cuh>
#include <humming/mma/umma.cuh>


template <MmaType kMmaType, class Ctx, class ArithClass>
struct MmaSelector;

template <class Ctx, class ArithClass>
struct MmaSelector<MmaType::MMA, Ctx, ArithClass> {
  using Type = WMMA<Ctx, ArithClass>;
};

template <class Ctx, class ArithClass>
struct MmaSelector<MmaType::WGMMA, Ctx, ArithClass> {
  using Type = WGMMA<Ctx, ArithClass>;
};

template <class Ctx, class ArithClass>
struct MmaSelector<MmaType::MXMMA, Ctx, ArithClass> {
  using Type = typename std::conditional<Ctx::kUseUmma, MXUMMA<Ctx, ArithClass>, MXMMA<Ctx, ArithClass>>::type;
};

template <class Ctx, class ArithClass>
struct MmaSelector<MmaType::UMMA, Ctx, ArithClass> {
  using Type = UMMA<Ctx, ArithClass>;
};

template <class Ctx, class ArithClass>
using Mma = typename MmaSelector<Ctx::kMmaType, Ctx, ArithClass>::Type;
