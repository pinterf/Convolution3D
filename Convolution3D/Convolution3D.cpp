/*

Convolution3D

Copyright (c) 2002 Sébastien LUCAS.  All rights reserved.
    babas.lucas@laposte.net


This file is subject to the terms of the GNU General Public License as
published by the Free Software Foundation.  A copy of this license is
included with this software distribution in the file COPYING.  If you
do not have a copy, you may obtain a copy by writing to the Free
Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details


*/

#if defined(_WIN32) && !defined(INTEL_INTRINSICS)
#error Forgot to set INTEL_INTRINSICS? Comment out this line if not
#endif

#ifdef _WIN32
#include "windows.h"
#endif
#include <string.h>
#include <stdio.h>
#include "avisynth.h"
#include "c3dcommon.h"  // common define

#ifdef INTEL_INTRINSICS
#include <emmintrin.h>
#endif

template<bool top, bool bottom>
AVS_FORCEINLINE void compute_fast121_c(
  const uint8_t* p_fcp, const uint8_t* p_fcc, const uint8_t* p_fcn,
  uint8_t* edi_dest,
  int pitch_p, int pitch_c, int pitch_n,
  int width,
  int thresh, int temp_thresh
)
{
  edi_dest[0] = p_fcc[0];
  for (int x = 1; x < width-1; x++) {

    int src = p_fcc[x];

    // current line: weight 2-4-2
    auto result = (src << 2);
    compute_mul_pixel_c(p_fcc[x - 1], 1, src, thresh, result);
    compute_mul_pixel_c(p_fcc[x + 1], 1, src, thresh, result);

    // prev line: weight 1-2-1
    auto prevline_p_fcc = p_fcc;
    if constexpr (!top)
      prevline_p_fcc -= pitch_c;
    compute_pixel_c(prevline_p_fcc[x - 1], src, thresh, result);
    compute_mul_pixel_c(prevline_p_fcc[x], 1, src, thresh, result);
    compute_pixel_c(prevline_p_fcc[x + 1], src, thresh, result);

    // next line: weight 1-2-1
    auto nextline_p_fcc = p_fcc;
    if constexpr (!bottom)
      nextline_p_fcc += pitch_c;
    compute_pixel_c(nextline_p_fcc[x - 1], src, thresh, result);
    compute_mul_pixel_c(nextline_p_fcc[x], 1, src, thresh, result);
    compute_pixel_c(nextline_p_fcc[x + 1], src, thresh, result);

    // actual result *= 2
    result <<= 1;

    // weight 16
    compute_mul_pixel_c(p_fcp[x], 4, src, temp_thresh, result);
    compute_mul_pixel_c(p_fcn[x], 4, src, temp_thresh, result);

    result = (result + 32) >> 6;
    edi_dest[x] = result;
  }
  edi_dest[width-1] = p_fcc[width - 1];
}

template<bool top, bool bottom>
AVS_FORCEINLINE void compute_121_c(
  const uint8_t* p_fcp, const uint8_t* p_fcc, const uint8_t* p_fcn,
  uint8_t* edi_dest,
  int pitch_p, int pitch_c, int pitch_n,
  int width,
  int thresh, int temp_thresh
)
{
  edi_dest[0] = p_fcc[0];
  for (int x = 1; x < width-1; x++) {

    int src = p_fcc[x];

    // current line: weight 2-4-2
    int result = (src << 2);
    compute_mul_pixel_c(p_fcc[x - 1], 1, src, thresh, result);
    compute_mul_pixel_c(p_fcc[x + 1], 1, src, thresh, result);

    // prev line: weight 1-2-1
    auto prevline_p_fcc = p_fcc;
    if constexpr (!top)
      prevline_p_fcc -= pitch_c;
    compute_pixel_c(prevline_p_fcc[x - 1], src, thresh, result);
    compute_mul_pixel_c(prevline_p_fcc[x], 1, src, thresh, result);
    compute_pixel_c(prevline_p_fcc[x + 1], src, thresh, result);

    // next line: weight 1-2-1
    auto nextline_p_fcc = p_fcc;
    if constexpr (!bottom)
      nextline_p_fcc += pitch_c;
    compute_pixel_c(nextline_p_fcc[x - 1], src, thresh, result);
    compute_mul_pixel_c(nextline_p_fcc[x], 1, src, thresh, result);
    compute_pixel_c(nextline_p_fcc[x + 1], src, thresh, result);

    // actual result *= 2
    result <<= 1;

    // temporal difference from 'fast': center*16 -> 1-2-1 prev, 2-4-2 current 1-2-1 next
    // previous
    compute_mul_pixel_c(p_fcp[x - 1], 1, src, temp_thresh, result);
    compute_mul_pixel_c(p_fcp[x], 2, src, temp_thresh, result);
    compute_mul_pixel_c(p_fcp[x + 1], 1, src, temp_thresh, result);
    // prev line: weight 1-2-1
    auto prevline_p_fcp = p_fcp;
    if constexpr (!top)
      prevline_p_fcp -= pitch_p;
    compute_pixel_c(prevline_p_fcp[x - 1], src, temp_thresh, result);
    compute_mul_pixel_c(prevline_p_fcp[x], 1, src, temp_thresh, result);
    compute_pixel_c(prevline_p_fcp[x + 1], src, temp_thresh, result);
    // next line: weight 1-2-1
    auto nextline_p_fcp = p_fcp;
    if constexpr (!bottom)
      nextline_p_fcp += pitch_p;
    compute_pixel_c(nextline_p_fcp[x - 1], src, temp_thresh, result);
    compute_mul_pixel_c(nextline_p_fcp[x], 1, src, temp_thresh, result);
    compute_pixel_c(nextline_p_fcp[x + 1], src, temp_thresh, result);

    // next
    compute_mul_pixel_c(p_fcn[x - 1], 1, src, temp_thresh, result);
    compute_mul_pixel_c(p_fcn[x], 2, src, temp_thresh, result);
    compute_mul_pixel_c(p_fcn[x + 1], 1, src, temp_thresh, result);
    // prev line: weight 1-2-1
    auto prevline_p_fcn = p_fcn;
    if constexpr (!top)
      prevline_p_fcn -= pitch_p;
    compute_pixel_c(prevline_p_fcn[x - 1], src, temp_thresh, result);
    compute_mul_pixel_c(prevline_p_fcn[x], 1, src, temp_thresh, result);
    compute_pixel_c(prevline_p_fcn[x + 1], src, temp_thresh, result);
    // next line: weight 1-2-1
    auto nextline_p_fcn = p_fcn;
    if constexpr (!bottom)
      nextline_p_fcn += pitch_p;
    compute_pixel_c(nextline_p_fcn[x - 1], src, temp_thresh, result);
    compute_mul_pixel_c(nextline_p_fcn[x], 1, src, temp_thresh, result);
    compute_pixel_c(nextline_p_fcn[x + 1], src, temp_thresh, result);

    result = (result + 32) >> 6;
    edi_dest[x] = result;
  }
  edi_dest[width - 1] = p_fcc[width - 1];
}


template<bool top, bool bottom>
AVS_FORCEINLINE void compute_fast111_c(
  const uint8_t* p_fcp, const uint8_t* p_fcc, const uint8_t* p_fcn,
  uint8_t* edi_dest,
  int pitch_p, int pitch_c, int pitch_n,
  int width,
  int thresh, int temp_thresh
)
{
  edi_dest[0] = p_fcc[0];
  for (int x = 1; x < width-1; x++) {

    int src = p_fcc[x];

    // current line: weight 1-1-1
    int result = src;
    compute_pixel_c(p_fcc[x - 1], src, thresh, result);
    compute_pixel_c(p_fcc[x + 1], src, thresh, result);

    // prev line: weight 1-1-1
    auto prevline_p_fcc = p_fcc;
    if constexpr (!top)
      prevline_p_fcc -= pitch_c;
    compute_pixel_c(prevline_p_fcc[x - 1], src, thresh, result);
    compute_pixel_c(prevline_p_fcc[x], src, thresh, result);
    compute_pixel_c(prevline_p_fcc[x + 1], src, thresh, result);

    // next line: weight 1-1-1
    auto nextline_p_fcc = p_fcc;
    if constexpr (!bottom)
      nextline_p_fcc += pitch_c;
    compute_pixel_c(nextline_p_fcc[x - 1], src, thresh, result);
    compute_pixel_c(nextline_p_fcc[x], src, thresh, result);
    compute_pixel_c(nextline_p_fcc[x + 1], src, thresh, result);

    // weight 1
    compute_pixel_c(p_fcp[x], src, temp_thresh, result);
    compute_pixel_c(p_fcn[x], src, temp_thresh, result);

    // 11 pixels; *2 for proper rounding 11/2 would not be nice
    //result = ((result << 1) + 11) / 22;
    result = (((result << 1) + 11) * (65536 / 22)) >> 16;
    edi_dest[x] = result;
  }
  edi_dest[width - 1] = p_fcc[width - 1];
}

template<bool top, bool bottom>
AVS_FORCEINLINE void compute_111_c(
  const uint8_t* p_fcp, const uint8_t* p_fcc, const uint8_t* p_fcn,
  uint8_t* edi_dest,
  int pitch_p, int pitch_c, int pitch_n,
  int width,
  int thresh, int temp_thresh
)
{
  edi_dest[0] = p_fcc[0];
  for (int x = 1; x < width-1; x++) {

    int src = p_fcc[x];

    // current line: weight 1-1-1
    int result = src;
    compute_pixel_c(p_fcc[x - 1], src, thresh, result);
    compute_pixel_c(p_fcc[x + 1], src, thresh, result);

    // prev line: weight 1-1-1
    auto prevline_p_fcc = p_fcc;
    if constexpr (!top)
      prevline_p_fcc -= pitch_c;
    compute_pixel_c(prevline_p_fcc[x - 1], src, thresh, result);
    compute_pixel_c(prevline_p_fcc[x], src, thresh, result);
    compute_pixel_c(prevline_p_fcc[x + 1], src, thresh, result);

    // next line: weight 1-1-1
    auto nextline_p_fcc = p_fcc;
    if constexpr (!bottom)
      nextline_p_fcc += pitch_c;
    compute_pixel_c(nextline_p_fcc[x - 1], src, thresh, result);
    compute_pixel_c(nextline_p_fcc[x], src, thresh, result);
    compute_pixel_c(nextline_p_fcc[x + 1], src, thresh, result);

    // temporal difference from 'fast': center*1 -> 1-1-1 prev, 1-1-1 current 1-1-1 next
    // previous
    compute_pixel_c(p_fcp[x - 1], src, temp_thresh, result);
    compute_pixel_c(p_fcp[x], src, temp_thresh, result);
    compute_pixel_c(p_fcp[x + 1], src, temp_thresh, result);
    // prev line: weight 1-2-1
    auto prevline_p_fcp = p_fcp;
    if constexpr (!top)
      prevline_p_fcp -= pitch_p;
    compute_pixel_c(prevline_p_fcp[x - 1], src, temp_thresh, result);
    compute_pixel_c(prevline_p_fcp[x], src, temp_thresh, result);
    compute_pixel_c(prevline_p_fcp[x + 1], src, temp_thresh, result);
    // next line: weight 1-2-1
    auto nextline_p_fcp = p_fcp;
    if constexpr (!bottom)
      nextline_p_fcp += pitch_p;
    compute_pixel_c(nextline_p_fcp[x - 1], src, temp_thresh, result);
    compute_pixel_c(nextline_p_fcp[x], src, temp_thresh, result);
    compute_pixel_c(nextline_p_fcp[x + 1], src, temp_thresh, result);

    // next
    compute_pixel_c(p_fcn[x - 1], src, temp_thresh, result);
    compute_pixel_c(p_fcn[x], src, temp_thresh, result);
    compute_pixel_c(p_fcn[x + 1], src, temp_thresh, result);
    // prev line: weight 1-2-1
    auto prevline_p_fcn = p_fcn;
    if constexpr (!top)
      prevline_p_fcn -= pitch_p;
    compute_pixel_c(prevline_p_fcn[x - 1], src, temp_thresh, result);
    compute_pixel_c(prevline_p_fcn[x], src, temp_thresh, result);
    compute_pixel_c(prevline_p_fcn[x + 1], src, temp_thresh, result);
    // next line: weight 1-2-1
    auto nextline_p_fcn = p_fcn;
    if constexpr (!bottom)
      nextline_p_fcn += pitch_p;
    compute_pixel_c(nextline_p_fcn[x - 1], src, temp_thresh, result);
    compute_pixel_c(nextline_p_fcn[x], src, temp_thresh, result);
    compute_pixel_c(nextline_p_fcn[x + 1], src, temp_thresh, result);

    // 27 pixels; *2 for proper rounding 27/2 would not be nice
    //result = ((result << 1) + 27) / 54;
    result = (((result << 1) + 27) * (65536/54)) >> 16;
    edi_dest[x] = result;
  }
  edi_dest[width - 1] = p_fcc[width - 1];
}

#ifdef INTEL_INTRINSICS
template<bool top, bool bottom>
AVS_FORCEINLINE void compute_fast121(
  const uint8_t* p_fcp, const uint8_t* p_fcc, const uint8_t* p_fcn,
  uint8_t* edi_dest,
  int pitch_p, int pitch_c, int pitch_n,
  int width,
  int thresh, int temp_thresh
)
{
  auto zero = _mm_setzero_si128();
  auto full_ff = _mm_set1_epi8(-1);
  auto round_32 = _mm_set1_epi16(32);

  for (int x = 0; x < width; x += 16) {

    auto src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p_fcc));

    // current line: weight 2-4-2
    // src*4
    auto mm4 = _mm_unpacklo_epi8(src, zero);
    auto mm5 = _mm_unpackhi_epi8(src, zero);
    mm4 = _mm_slli_epi16(mm4, 2);
    mm5 = _mm_slli_epi16(mm5, 2);

    auto simd_thresh = _mm_set1_epi8(thresh);
    // mm4, mm5 input output
    compute_mul_pixel(p_fcc - 1, 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_mul_pixel(p_fcc + 1, 1, full_ff, src, simd_thresh, zero, mm4, mm5);

    // prev line: weight 1-2-1
    auto prevline_p_fcc = p_fcc;
    if constexpr (!top)
      prevline_p_fcc -= pitch_c;
    compute_pixel(prevline_p_fcc - 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_mul_pixel(prevline_p_fcc, 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_pixel(prevline_p_fcc + 1, full_ff, src, simd_thresh, zero, mm4, mm5);

    // next line: weight 1-2-1
    auto nextline_p_fcc = p_fcc;
    if constexpr (!bottom)
      nextline_p_fcc += pitch_c;
    compute_pixel(nextline_p_fcc - 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_mul_pixel(nextline_p_fcc, 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_pixel(nextline_p_fcc + 1, full_ff, src, simd_thresh, zero, mm4, mm5);

    p_fcc += 16;

    // actual result *= 2
    mm4 = _mm_slli_epi16(mm4, 1);
    mm5 = _mm_slli_epi16(mm5, 1);

    auto simd_temp_thresh = _mm_set1_epi8(temp_thresh);

    // weight 16
    compute_mul_pixel(p_fcp, 4, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    p_fcp += 16;
    compute_mul_pixel(p_fcn, 4, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    p_fcn += 16;

    mm4 = _mm_add_epi16(mm4, round_32);
    mm5 = _mm_add_epi16(mm5, round_32);
    mm4 = _mm_srli_epi16(mm4, 6); // /64
    mm5 = _mm_srli_epi16(mm5, 6);
    mm4 = _mm_packus_epi16(mm4, mm5);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(edi_dest), mm4);
    edi_dest += 16;
  }
}

template<bool top, bool bottom>
AVS_FORCEINLINE void compute_121(
  const uint8_t* p_fcp, const uint8_t* p_fcc, const uint8_t* p_fcn,
  uint8_t* edi_dest,
  int pitch_p, int pitch_c, int pitch_n,
  int width,
  int thresh, int temp_thresh
)
{
  auto zero = _mm_setzero_si128();
  auto full_ff = _mm_set1_epi8(-1);
  auto round_32 = _mm_set1_epi16(32);

  for (int x = 0; x < width; x += 16) {

    auto src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p_fcc));

    // current line: weight 2-4-2
    // src*4
    auto mm4 = _mm_unpacklo_epi8(src, zero);
    auto mm5 = _mm_unpackhi_epi8(src, zero);
    mm4 = _mm_slli_epi16(mm4, 2);
    mm5 = _mm_slli_epi16(mm5, 2);

    auto simd_thresh = _mm_set1_epi8(thresh);
    // mm4, mm5 input output
    compute_mul_pixel(p_fcc - 1, 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_mul_pixel(p_fcc + 1, 1, full_ff, src, simd_thresh, zero, mm4, mm5);

    // prev line: weight 1-2-1
    auto prevline_p_fcc = p_fcc;
    if constexpr (!top)
      prevline_p_fcc -= pitch_c;
    compute_pixel(prevline_p_fcc - 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_mul_pixel(prevline_p_fcc, 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_pixel(prevline_p_fcc + 1, full_ff, src, simd_thresh, zero, mm4, mm5);

    // next line: weight 1-2-1
    auto nextline_p_fcc = p_fcc;
    if constexpr (!bottom)
      nextline_p_fcc += pitch_c;
    compute_pixel(nextline_p_fcc - 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_mul_pixel(nextline_p_fcc, 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_pixel(nextline_p_fcc + 1, full_ff, src, simd_thresh, zero, mm4, mm5);

    p_fcc += 16;

    // actual result *= 2
    mm4 = _mm_slli_epi16(mm4, 1);
    mm5 = _mm_slli_epi16(mm5, 1);

    auto simd_temp_thresh = _mm_set1_epi8(temp_thresh); // movq mm6, temp_thresh_mask

    // temporal difference from 'fast': center*16 -> 1-2-1 prev, 2-4-2 current 1-2-1 next
    compute_mul_pixel(p_fcp - 1, 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_mul_pixel(p_fcp, 2, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_mul_pixel(p_fcp + 1, 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    // prev line: weight 1-2-1
    auto prevline_p_fcp = p_fcp;
    if constexpr (!top)
      prevline_p_fcp -= pitch_p;
    compute_pixel(prevline_p_fcp - 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_mul_pixel(prevline_p_fcp, 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(prevline_p_fcp + 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    // next line: weight 1-2-1
    auto nextline_p_fcp = p_fcp;
    if constexpr (!bottom)
      nextline_p_fcp += pitch_p;
    compute_pixel(nextline_p_fcp - 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_mul_pixel(nextline_p_fcp, 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(nextline_p_fcp + 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);

    p_fcp += 16;

    compute_mul_pixel(p_fcn - 1, 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_mul_pixel(p_fcn, 2, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_mul_pixel(p_fcn + 1, 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    // prev line: weight 1-2-1
    auto prevline_p_fcn = p_fcn;
    if constexpr (!top)
      prevline_p_fcn -= pitch_n;
    compute_pixel(prevline_p_fcn - 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_mul_pixel(prevline_p_fcn, 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(prevline_p_fcn + 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    // next line: weight 1-2-1
    auto nextline_p_fcn = p_fcn;
    if constexpr (!bottom)
      nextline_p_fcn += pitch_n;
    compute_pixel(nextline_p_fcn - 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_mul_pixel(nextline_p_fcn, 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(nextline_p_fcn + 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);

    p_fcn += 16;

    mm4 = _mm_add_epi16(mm4, round_32);
    mm5 = _mm_add_epi16(mm5, round_32);
    mm4 = _mm_srli_epi16(mm4, 6); // /64
    mm5 = _mm_srli_epi16(mm5, 6);
    mm4 = _mm_packus_epi16(mm4, mm5);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(edi_dest), mm4);
    edi_dest += 16;
  }
}

template<bool top, bool bottom>
AVS_FORCEINLINE void compute_fast111(
  const uint8_t* p_fcp, const uint8_t* p_fcc, const uint8_t* p_fcn,
  uint8_t* edi_dest,
  int pitch_p, int pitch_c, int pitch_n,
  int width,
  int thresh, int temp_thresh
)
{
  auto zero = _mm_setzero_si128();
  auto full_ff = _mm_set1_epi8(-1);
  auto round_11 = _mm_set1_epi16(11);
  auto multi_11 = _mm_set1_epi16((65536 / 2) / 11);

  for (int x = 0; x < width; x += 16) {

    auto src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p_fcc));

    // current line: weight 1-1-1
    auto mm4 = _mm_unpacklo_epi8(src, zero);
    auto mm5 = _mm_unpackhi_epi8(src, zero);

    auto simd_thresh = _mm_set1_epi8(thresh);
    // mm4, mm5 input output
    compute_pixel(p_fcc - 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_pixel(p_fcc + 1, full_ff, src, simd_thresh, zero, mm4, mm5);

    // prev line: weight 1-1-1
    auto prevline_p_fcc = p_fcc;
    if constexpr (!top)
      prevline_p_fcc -= pitch_c;
    compute_pixel(prevline_p_fcc - 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_pixel(prevline_p_fcc, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_pixel(prevline_p_fcc + 1, full_ff, src, simd_thresh, zero, mm4, mm5);

    // next line: weight 1-1-1
    auto nextline_p_fcc = p_fcc;
    if constexpr (!bottom)
      nextline_p_fcc += pitch_c;
    compute_pixel(nextline_p_fcc - 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_pixel(nextline_p_fcc, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_pixel(nextline_p_fcc + 1, full_ff, src, simd_thresh, zero, mm4, mm5);

    p_fcc += 16;

    auto simd_temp_thresh = _mm_set1_epi8(temp_thresh);

    // weight 1
    compute_pixel(p_fcp, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    p_fcp += 16;
    compute_pixel(p_fcn, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    p_fcn += 16;

    // 11 pixels; *2 for proper rounding 11/2 would not be nice

    // actual result *= 2
    mm4 = _mm_slli_epi16(mm4, 1);
    mm5 = _mm_slli_epi16(mm5, 1);
    mm4 = _mm_add_epi16(mm4, round_11);
    mm5 = _mm_add_epi16(mm5, round_11);
    mm4 = _mm_mulhi_epu16(mm4, multi_11); // (65536/2) / 11
    mm5 = _mm_mulhi_epu16(mm5, multi_11);
    mm4 = _mm_packus_epi16(mm4, mm5);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(edi_dest), mm4);
    edi_dest += 16;
  }
}

// pl_pitch_p and pl_pitch_n is zero for top line
// nl_pitch_p and nl_pitch_n is zero for bottom line
template<bool top, bool bottom>
AVS_FORCEINLINE void compute_111(
  const uint8_t* p_fcp, const uint8_t* p_fcc, const uint8_t* p_fcn,
  uint8_t* edi_dest,
  int pitch_p, int pitch_c, int pitch_n,
  int width,
  int thresh, int temp_thresh
)
{
  auto zero = _mm_setzero_si128();
  auto full_ff = _mm_set1_epi8(-1);
  auto round_27 = _mm_set1_epi16(27);
  auto multi_27 = _mm_set1_epi16((65536 / 2) / 27);

  for (int x = 0; x < width; x += 16) {

    auto src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p_fcc));

    // current line: weight 1-1-1
    auto mm4 = _mm_unpacklo_epi8(src, zero);
    auto mm5 = _mm_unpackhi_epi8(src, zero);

    auto simd_thresh = _mm_set1_epi8(thresh);
    // mm4, mm5 input output
    compute_pixel(p_fcc - 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_pixel(p_fcc + 1, full_ff, src, simd_thresh, zero, mm4, mm5);

    // prev line: weight 1-1-1
    auto prevline_p_fcc = p_fcc;
    if constexpr (!top)
      prevline_p_fcc -= pitch_c;

    compute_pixel(prevline_p_fcc - 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_pixel(prevline_p_fcc, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_pixel(prevline_p_fcc + 1, full_ff, src, simd_thresh, zero, mm4, mm5);

    // next line: weight 1-1-1
    auto nextline_p_fcc = p_fcc;
    if constexpr (!bottom)
      nextline_p_fcc += pitch_c;
    compute_pixel(nextline_p_fcc - 1, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_pixel(nextline_p_fcc, full_ff, src, simd_thresh, zero, mm4, mm5);
    compute_pixel(nextline_p_fcc + 1, full_ff, src, simd_thresh, zero, mm4, mm5);

    p_fcc += 16;

    auto simd_temp_thresh = _mm_set1_epi8(temp_thresh);

    // temporal difference from 'fast': center*1 -> 1-1-1 prev, 1-1-1 current 1-1-1 next
    compute_pixel(p_fcp - 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(p_fcp, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(p_fcp + 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    // prev line: weight 1-1-1
    auto prevline_p_fcp = p_fcp;
    if constexpr (!top)
      prevline_p_fcp -= pitch_p;
    compute_pixel(prevline_p_fcp - 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(prevline_p_fcp, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(prevline_p_fcp + 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    // next line: weight 1-1-1
    auto nextline_p_fcp = p_fcp;
    if constexpr (!bottom)
      nextline_p_fcp += pitch_p;
    compute_pixel(nextline_p_fcp - 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(nextline_p_fcp, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(nextline_p_fcp + 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);

    p_fcp += 16;

    compute_pixel(p_fcn - 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(p_fcn, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(p_fcn + 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    // prev line: weight 1-1-1
    auto prevline_p_fcn = p_fcn;
    if constexpr (!top)
      prevline_p_fcn -= pitch_n;
    compute_pixel(prevline_p_fcn - 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(prevline_p_fcn, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(prevline_p_fcn + 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    // next line: weight 1-1-1
    auto nextline_p_fcn = p_fcn;
    if constexpr (!bottom)
      nextline_p_fcn += pitch_n;
    compute_pixel(nextline_p_fcn - 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(nextline_p_fcn, full_ff, src, simd_temp_thresh, zero, mm4, mm5);
    compute_pixel(nextline_p_fcn + 1, full_ff, src, simd_temp_thresh, zero, mm4, mm5);

    p_fcn += 16;

    // 27 pixels; *2 for proper rounding 27/2 would not be nice

    // actual result *= 2
    mm4 = _mm_slli_epi16(mm4, 1);
    mm5 = _mm_slli_epi16(mm5, 1);
    mm4 = _mm_add_epi16(mm4, round_27);
    mm5 = _mm_add_epi16(mm5, round_27);
    mm4 = _mm_mulhi_epu16(mm4, multi_27); // (65536/2) / 27
    mm5 = _mm_mulhi_epu16(mm5, multi_27);
    mm4 = _mm_packus_epi16(mm4, mm5);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(edi_dest), mm4);
    edi_dest += 16;
  }
}
#endif // #ifdef INTEL_INTRINSICS

using proc_fn_t = decltype(compute_111_c<false, false>);

template<proc_fn_t processor_function_top, proc_fn_t processor_function, proc_fn_t processor_function_bottom>
void process_c(
  const unsigned char* p_fcp, int pitch_p,
  const unsigned char* p_fcc, int pitch_c,
  const unsigned char* p_fcn, int pitch_n,
  unsigned char* p_dest, int pitch_d,
  int width, int height,
  int temp_thresh_mask,
  int thresh_mask)
{
  // top
  processor_function_top(p_fcp, p_fcc, p_fcn, p_dest, pitch_p, pitch_c, pitch_n, width, thresh_mask, temp_thresh_mask);
  p_fcp += pitch_p;
  p_fcc += pitch_c;
  p_fcn += pitch_n;
  p_dest += pitch_d;
  for (int y = 1; y < height - 1; y++) {
    processor_function(p_fcp, p_fcc, p_fcn, p_dest, pitch_p, pitch_c, pitch_n, width, thresh_mask, temp_thresh_mask);
    p_fcp += pitch_p;
    p_fcc += pitch_c;
    p_fcn += pitch_n;
    p_dest += pitch_d;
  }
  processor_function_bottom(p_fcp, p_fcc, p_fcn, p_dest, pitch_p, pitch_c, pitch_n, width, thresh_mask, temp_thresh_mask);
}

#ifdef INTEL_INTRINSICS
template<proc_fn_t processor_function_top, proc_fn_t processor_function, proc_fn_t processor_function_bottom>
void process_simd(
  const unsigned char* p_fcp, int pitch_p,
  const unsigned char* p_fcc, int pitch_c,
  const unsigned char* p_fcn, int pitch_n,
  unsigned char* p_dest, int pitch_d,
  int width, int height,
  int temp_thresh_mask,
  int thresh_mask)
{
  // top
  const int offset = width - 16 - 1;

  p_dest[0] = p_fcc[0];
  processor_function_top(p_fcp + 1, p_fcc + 1, p_fcn + 1, p_dest + 1, pitch_p, pitch_c, pitch_n, 16, thresh_mask, temp_thresh_mask); // leftmost but one 16 pixels, overlaps 1 pixel with safe middle area
  processor_function_top(p_fcp + 16, p_fcc + 16, p_fcn + 16, p_dest + 16, pitch_p, pitch_c, pitch_n, width - 2 * 16, thresh_mask, temp_thresh_mask); // middle safe
  processor_function_top(p_fcp + offset, p_fcc + offset, p_fcn + offset, p_dest + offset, pitch_p, pitch_c, pitch_n, 16, thresh_mask, temp_thresh_mask); // last but one 16 pixels
  p_dest[width - 1] = p_fcc[width - 1]; // last pixel

  p_fcp += pitch_p;
  p_fcc += pitch_c;
  p_fcn += pitch_n;
  p_dest += pitch_d;
  for (int y = 1; y < height - 1; y++) {
    p_dest[0] = p_fcc[0];
    processor_function(p_fcp + 1, p_fcc + 1, p_fcn + 1, p_dest + 1, pitch_p, pitch_c, pitch_n, 16, thresh_mask, temp_thresh_mask);
    processor_function(p_fcp + 16, p_fcc + 16, p_fcn + 16, p_dest + 16, pitch_p, pitch_c, pitch_n, width - 2 * 16, thresh_mask, temp_thresh_mask);
    processor_function(p_fcp + offset, p_fcc + offset, p_fcn + offset, p_dest + offset, pitch_p, pitch_c, pitch_n, 16, thresh_mask, temp_thresh_mask);
    p_dest[width - 1] = p_fcc[width - 1];
    p_fcp += pitch_p;
    p_fcc += pitch_c;
    p_fcn += pitch_n;
    p_dest += pitch_d;
  }
  p_dest[0] = p_fcc[0];
  processor_function_bottom(p_fcp + 1, p_fcc + 1, p_fcn + 1, p_dest + 1, pitch_p, pitch_c, pitch_n, 16, thresh_mask, temp_thresh_mask);
  processor_function_bottom(p_fcp + 16, p_fcc + 16, p_fcn + 16, p_dest + 16, pitch_p, pitch_c, pitch_n, width - 2 * 16, thresh_mask, temp_thresh_mask);
  processor_function_bottom(p_fcp + offset, p_fcc + offset, p_fcn + offset, p_dest + offset, pitch_p, pitch_c, pitch_n, 16, thresh_mask, temp_thresh_mask);
  p_dest[width - 1] = p_fcc[width - 1];
}
#endif

#ifdef INTEL_INTRINSICS
// instantiate
template void process_simd<compute_111<true, false>, compute_111<false, false>, compute_111<false, true>>(const unsigned char* p_fcp, int pitch_p, const unsigned char* p_fcc, int pitch_c, const unsigned char* p_fcn, int pitch_n, unsigned char* p_dest, int pitch_d, int width, int height, int temp_thresh_mask, int thresh_mask);
template void process_simd<compute_fast111<true, false>, compute_fast111<false, false>, compute_fast111<false, true>>(const unsigned char* p_fcp, int pitch_p, const unsigned char* p_fcc, int pitch_c, const unsigned char* p_fcn, int pitch_n, unsigned char* p_dest, int pitch_d, int width, int height, int temp_thresh_mask, int thresh_mask);
template void process_simd<compute_121<true, false>, compute_121<false, false>, compute_121<false, true>>(const unsigned char* p_fcp, int pitch_p, const unsigned char* p_fcc, int pitch_c, const unsigned char* p_fcn, int pitch_n, unsigned char* p_dest, int pitch_d, int width, int height, int temp_thresh_mask, int thresh_mask);
template void process_simd<compute_fast121<true, false>, compute_fast121<false, false>, compute_fast121<false, true>>(const unsigned char* p_fcp, int pitch_p, const unsigned char* p_fcc, int pitch_c, const unsigned char* p_fcn, int pitch_n, unsigned char* p_dest, int pitch_d, int width, int height, int temp_thresh_mask, int thresh_mask);
#endif

template void process_c<compute_111_c<true, false>, compute_111_c<false, false>, compute_111_c<false, true>>(const unsigned char* p_fcp, int pitch_p, const unsigned char* p_fcc, int pitch_c, const unsigned char* p_fcn, int pitch_n, unsigned char* p_dest, int pitch_d, int width, int height, int temp_thresh_mask, int thresh_mask);
template void process_c<compute_fast111_c<true, false>, compute_fast111_c<false, false>, compute_fast111_c<false, true>>(const unsigned char* p_fcp, int pitch_p, const unsigned char* p_fcc, int pitch_c, const unsigned char* p_fcn, int pitch_n, unsigned char* p_dest, int pitch_d, int width, int height, int temp_thresh_mask, int thresh_mask);
template void process_c<compute_121_c<true, false>, compute_121_c<false, false>, compute_121_c<false, true>>(const unsigned char* p_fcp, int pitch_p, const unsigned char* p_fcc, int pitch_c, const unsigned char* p_fcn, int pitch_n, unsigned char* p_dest, int pitch_d, int width, int height, int temp_thresh_mask, int thresh_mask);
template void process_c<compute_fast121_c<true, false>, compute_fast121_c<false, false>, compute_fast121_c<false, true>>(const unsigned char* p_fcp, int pitch_p, const unsigned char* p_fcc, int pitch_c, const unsigned char* p_fcn, int pitch_n, unsigned char* p_dest, int pitch_d, int width, int height, int temp_thresh_mask, int thresh_mask);

using frameproc_fn_t = decltype(process_c<compute_fast121_c<true, false>, compute_fast121_c<false, false>, compute_fast121_c<false, true>>);

class Convolution3D : public GenericVideoFilter
{
  bool has_at_least_v8;

  int debug;

  frameproc_fn_t *funcPtr_simd;
  frameproc_fn_t* funcPtr_c;

  frameproc_fn_t* funcPtr_luma;
  frameproc_fn_t* funcPtr_chroma;

  bool copyLuma, copyChroma;
  int luma_Treshold, chroma_Treshold;
  int temporal_luma_Treshold, temporal_chroma_Treshold;
  double temporal_influence;
  int luma_limit;
  int matrix;
  int opt;

public:
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  Convolution3D(PClip _child, short _matrix, short _luma_Treshold, short _chroma_Treshold,
    short _temporal_luma_Treshold, short _temporal_chroma_Treshold,
    double _temporal_influence, int _debug, int _opt, IScriptEnvironment* env)
    : GenericVideoFilter(_child), matrix(_matrix), luma_Treshold(_luma_Treshold), chroma_Treshold(_chroma_Treshold),
    temporal_luma_Treshold(_temporal_luma_Treshold), temporal_chroma_Treshold(_temporal_chroma_Treshold),
    temporal_influence(_temporal_influence), debug(_debug), opt(_opt)
  {
    char dbgString[100];

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }

    if (!vi.IsY8() && !vi.IsYV12() && !vi.IsYV16() && !vi.IsYV411() && !vi.IsYV24())
      env->ThrowError("Convolution3D supports Y8, YV12, YV411, YV16 or YV24 color format only");

#ifdef INTEL_INTRINSICS
    if (!(env->GetCPUFlags() & CPUF_SSE2))
      env->ThrowError("Convolution3D needs a SSE2 capable CPU");
#endif

    if (matrix < 0 || matrix > 3)
      env->ThrowError("Convolution3D : matrix parameter must be 0, 1, 2 or 3");

    if (luma_Treshold < MIN_TRESHOLD || luma_Treshold > MAX_TRESHOLD ||
      chroma_Treshold < MIN_TRESHOLD || chroma_Treshold > MAX_TRESHOLD ||
      temporal_luma_Treshold < MIN_TRESHOLD || temporal_luma_Treshold > MAX_TRESHOLD ||
      temporal_chroma_Treshold < MIN_TRESHOLD || temporal_chroma_Treshold > MAX_TRESHOLD)
      env->ThrowError("Convolution3D : all tresholds must be between 0 and 255");

    if (temporal_influence < MIN_TEMPORAL_INFLUENCE || temporal_influence > MAX_TEMPORAL_INFLUENCE)
      env->ThrowError("Convolution3D : temporal influence must be between -1 and 100");

    luma_limit = (int)(temporal_luma_Treshold * temporal_influence);
    switch (matrix)
    {
    case STANDARD_MATRIX:
#ifdef INTEL_INTRINSICS
      funcPtr_simd = &process_simd<compute_121<true, false>, compute_121<false, false>, compute_121<false, true>>;
#endif
      funcPtr_c = &process_c<compute_121_c<true, false>, compute_121_c<false, false>, compute_121_c<false, true>>;
      break;
    case SIMPLE_MATRIX:
#ifdef INTEL_INTRINSICS
      funcPtr_simd = &process_simd<compute_111<true, false>, compute_111<false, false>, compute_111<false, true>>;
#endif
      funcPtr_c = &process_c<compute_111_c<true, false>, compute_111_c<false, false>, compute_111_c<false, true>>;
      break;
    case STANDARD_FAST_MATRIX:
#ifdef INTEL_INTRINSICS
      funcPtr_simd = &process_simd<compute_fast121<true, false>, compute_fast121<false, false>, compute_fast121<false, true>>;
#endif
      funcPtr_c = &process_c<compute_fast121_c<true, false>, compute_fast121_c<false, false>, compute_fast121_c<false, true>>;
      break;
    case SIMPLE_FAST_MATRIX:
#ifdef INTEL_INTRINSICS
      funcPtr_simd = &process_simd<compute_fast111<true, false>, compute_fast111<false, false>, compute_fast111<false, true>>;
#endif
      funcPtr_c = &process_c<compute_fast111_c<true, false>, compute_fast111_c<false, false>, compute_fast111_c<false, true>>;
      break;
    }

    funcPtr_luma = funcPtr_c;
    funcPtr_chroma = vi.IsY() ? nullptr : funcPtr_c;


#ifdef INTEL_INTRINSICS
    bool use_SSE2 = (env->GetCPUFlags() & CPUF_SSE2) == CPUF_SSE2;
    if (opt == 0) use_SSE2 = false;

    if (use_SSE2) {
      if (vi.width >= 16 + 2)
        funcPtr_luma = funcPtr_simd;
      if (!vi.IsY() && vi.IsYUV()) {
        if ((vi.width >> vi.GetPlaneWidthSubsampling(PLANAR_U)) >= 16 + 2)
          funcPtr_chroma = funcPtr_simd;
      }
    }
#endif

    copyChroma = (chroma_Treshold == 0) && (temporal_chroma_Treshold == 0);
    copyLuma = (luma_Treshold == 0) && (temporal_luma_Treshold == 0);

    if (debug)
    {
      sprintf(dbgString, "Convolution3D for Avisynth\n");
#ifdef _WIN32
      OutputDebugString(dbgString);
#endif
    }
    SetCacheHints(CACHE_WINDOW, 2);

  };

private:

};


PVideoFrame __stdcall Convolution3D::GetFrame(int n, IScriptEnvironment* env)
{
  // c: current p: previous n: next
  PVideoFrame fc = child->GetFrame(n, env);
  PVideoFrame fp = child->GetFrame(n == 0 ? 0 : n - 1, env);
  PVideoFrame fn = child->GetFrame(n >= vi.num_frames - 1 ? vi.num_frames - 1 : n + 1, env);
  PVideoFrame final = has_at_least_v8 ? env->NewVideoFrameP(vi, &fc) : env->NewVideoFrame(vi);


  if (copyLuma)
    env->BitBlt(final->GetWritePtr(PLANAR_Y), final->GetPitch(PLANAR_Y),
      fc->GetReadPtr(PLANAR_Y), fc->GetPitch(PLANAR_Y),
      fc->GetRowSize(PLANAR_Y), fc->GetHeight(PLANAR_Y));
  else
    funcPtr_luma(fp->GetReadPtr(PLANAR_Y), fp->GetPitch(PLANAR_Y),
      fc->GetReadPtr(PLANAR_Y), fc->GetPitch(PLANAR_Y),
      fn->GetReadPtr(PLANAR_Y), fn->GetPitch(PLANAR_Y),
      final->GetWritePtr(PLANAR_Y), final->GetPitch(PLANAR_Y),
      vi.width, fc->GetHeight(PLANAR_Y),
      temporal_luma_Treshold,
      luma_Treshold);

  if (copyChroma)
  {
    env->BitBlt(final->GetWritePtr(PLANAR_U), final->GetPitch(PLANAR_U),
      fc->GetReadPtr(PLANAR_U), fc->GetPitch(PLANAR_U),
      fc->GetRowSize(PLANAR_U), fc->GetHeight(PLANAR_U));
    env->BitBlt(final->GetWritePtr(PLANAR_V), final->GetPitch(PLANAR_V),
      fc->GetReadPtr(PLANAR_V), fc->GetPitch(PLANAR_V),
      fc->GetRowSize(PLANAR_V), fc->GetHeight(PLANAR_V));
  }
  else if(funcPtr_chroma != nullptr)
  {
    const int cwidth = vi.width >> vi.GetPlaneWidthSubsampling(PLANAR_U);
    funcPtr_chroma(fp->GetReadPtr(PLANAR_U), fp->GetPitch(PLANAR_U),
      fc->GetReadPtr(PLANAR_U), fc->GetPitch(PLANAR_U),
      fn->GetReadPtr(PLANAR_U), fn->GetPitch(PLANAR_U),
      final->GetWritePtr(PLANAR_U), final->GetPitch(PLANAR_U),
      cwidth, fc->GetHeight(PLANAR_U),
      temporal_chroma_Treshold,
      chroma_Treshold);

    funcPtr_chroma(fp->GetReadPtr(PLANAR_V), fp->GetPitch(PLANAR_V),
      fc->GetReadPtr(PLANAR_V), fc->GetPitch(PLANAR_V),
      fn->GetReadPtr(PLANAR_V), fn->GetPitch(PLANAR_V),
      final->GetWritePtr(PLANAR_V), final->GetPitch(PLANAR_V),
      cwidth, fc->GetHeight(PLANAR_V),
      temporal_chroma_Treshold,
      chroma_Treshold);
  }
  return final;
}


AVSValue __cdecl Create_Convolution3D(AVSValue args, void* user_data, IScriptEnvironment* env) {
  return new Convolution3D(args[0].AsClip(), args[1].AsInt(0), // Matrix choice
    args[2].AsInt(3), args[3].AsInt(4),  // Spatial treshold
    args[4].AsInt(3), args[5].AsInt(4),  // Temporal treshold
    args[6].AsFloat(3),
    args[7].AsInt(0), // debug
    args[8].AsInt(-1), // opt
    env);
}

AVSValue __cdecl Create_Convolution3D_Pre(AVSValue args, void* user_data, IScriptEnvironment* env) {
  const char* myString = args[1].AsString("");
  if (!_stricmp(myString, "movieHQ"))
  {
    return new Convolution3D(args[0].AsClip(), 0, 3, 4, 3, 4, 2.8, 0, -1, env);
  }
  else if (!_stricmp(myString, "movieLQ"))
  {
    return new Convolution3D(args[0].AsClip(), 0, 6, 10, 6, 8, 2.8, 0, -1, env);
  }
  else if (!_stricmp(myString, "animeHQ"))
  {
    return new Convolution3D(args[0].AsClip(), 0, 6, 12, 6, 8, 2.8, 0, -1, env);
  }
  else if (!_stricmp(myString, "animeLQ"))
  {
    return new Convolution3D(args[0].AsClip(), 1, 8, 16, 8, 8, 2.8, 0, -1, env);
  }
  else if (!_stricmp(myString, "animeBQ"))
  {
    return new Convolution3D(args[0].AsClip(), 1, 12, 22, 8, 8, 2.8, 0, -1, env);
  }
  else if (!_stricmp(myString, "vhsBQ"))
  {
    return new Convolution3D(args[0].AsClip(), 0, 16, 48, 10, 32, 4, 0, -1, env);
  }
  else
  {
    env->ThrowError("Correct preset values are : movie[HQ, LQ], anime[HQ, LQ, BQ] or vhs[BQ]");
  }
  return new Convolution3D(args[0].AsClip(), 1, 6, 12, 6, 8, 2.8, 0, -1, env);
}


const AVS_Linkage* AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
  AVS_linkage = vectors;

  env->AddFunction("Convolution3D", "c[matrix]i[ythresh]i[cthresh]i[t_ythresh]i[t_cthresh]i[influence]f[debug]i[opt]i", Create_Convolution3D, 0);
  env->AddFunction("Convolution3D", "c[preset]si", Create_Convolution3D_Pre, 0);
  return "`Convolution3D' Denoiser";
}
