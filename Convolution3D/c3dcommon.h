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

// 1 2 1 
#define STANDARD_MATRIX 0
// 1 1 1
#define SIMPLE_MATRIX	1
// 1 2 1 
#define STANDARD_FAST_MATRIX 2
// 1 1 1
#define SIMPLE_FAST_MATRIX	3


#define MIN_TRESHOLD    0
#define MAX_TRESHOLD	255
#define MIN_TEMPORAL_INFLUENCE -1
#define MAX_TEMPORAL_INFLUENCE 100

#include <emmintrin.h>
#include <algorithm>

int check_bthreshold_c(int orig, int pixel_mask, int thresh)
{
  auto absdiff = std::abs(orig - pixel_mask);
  if (absdiff <= thresh)
    return orig;
  return pixel_mask;
}

// check treshold on byte
// What it does is compute an absolute difference 
// between the current pixel and the pixel mask
// Next we substract the treshold -> all values = 0 are good and
// must be kept, others must be replaced by the mask
AVS_FORCEINLINE __m128i check_bthreshold(__m128i orig, __m128i full_ff, __m128i pixel_mask, __m128i thresh, __m128i zero)
{
  auto absdiff = _mm_subs_epu8(_mm_max_epu8(orig, pixel_mask), _mm_min_epu8(orig, pixel_mask)); // abs diff
  auto belowthresh = _mm_subs_epu8(absdiff, thresh);
  auto comp_res = _mm_cmpeq_epi8(belowthresh, zero); // FF where absdiff <= thresh
  // blend
  auto mm1 = _mm_andnot_si128(comp_res, pixel_mask); // pixel_mask where comp_res false (00)
  auto mm0 = _mm_and_si128(comp_res, orig); // orig where comp_res is true (FF)
  return _mm_or_si128(mm0, mm1);
}

AVS_FORCEINLINE void compute_mul_pixel_c(int orig, int mul, int pixel_mask, int thresh, int& result)
{
  auto res = check_bthreshold_c(orig, pixel_mask, thresh);
  result += res << mul;
}

AVS_FORCEINLINE void compute_mul_pixel(const uint8_t *where, int mul,
  __m128i full_ff, __m128i center, __m128i thresh, __m128i zero, __m128i& result_lo, __m128i& result_hi)
{
  auto mm0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(where));
  auto res = check_bthreshold(mm0, full_ff, center, thresh, zero);
  auto mm1 = _mm_unpacklo_epi8(res, zero);
  auto mm2 = _mm_unpackhi_epi8(res, zero);
  mm1 = _mm_slli_epi16(mm1, mul);
  mm2 = _mm_slli_epi16(mm2, mul);
  result_lo = _mm_add_epi16(result_lo, mm1);
  result_hi = _mm_add_epi16(result_hi, mm2);
}

AVS_FORCEINLINE void compute_pixel_c(int orig, int center, int thresh, int& result)
{
  auto res = check_bthreshold_c(orig, center, thresh);
  result += res;
}

// same as compute_mul_pixel but no shift inside
AVS_FORCEINLINE void compute_pixel(const uint8_t* where,
  __m128i full_ff, __m128i center, __m128i thresh, __m128i zero, __m128i& result_lo, __m128i& result_hi)
{
  auto mm0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(where));
  auto res = check_bthreshold(mm0, full_ff, center, thresh, zero);
  auto mm1 = _mm_unpacklo_epi8(res, zero);
  auto mm2 = _mm_unpackhi_epi8(res, zero);
  result_lo = _mm_add_epi16(result_lo, mm1);
  result_hi = _mm_add_epi16(result_hi, mm2);
}
