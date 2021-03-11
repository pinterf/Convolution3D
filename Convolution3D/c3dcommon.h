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

#if 0
  // Check treshold on word
  // The value to compute must be in mm0 and the result will be in mm1
  // mm7 = upper limit
  // mm6 = lower limit
  // pixel_mask = default value
__m128i check_threshold(__m128i orig, __m128i& full_f, __m128i& pixel_mask, __m128i &mm6, __m128i& mm7)
{
  auto mm2 = orig;
  auto mm0 = _mm_max_epi16(orig, mm6);
  mm0 = _mm_min_epi16(mm0, mm7);
  mm0 = _mm_cmpeq_epi16(mm0, mm2);
  auto mm1 = mm0;
  mm0 = _mm_andnot_si128(mm0, pixel_mask);
  mm1 = _mm_and_si128(mm1, mm2);
  mm1 = _mm_or_si128(mm1, mm0);
  return mm1;
}

// Check treshold on word
// FIXME PF: seems it's not word but byte! never mind, not used.
// The value to compute must be in mm0 and the result will be in mm1
// mm7 = upper limit
// mm6 = lower limit
// pixel_mask = default value
__m128i check_lthreshold(__m128i& mm0, __m128i& full_f, __m128i& pixel_mask, __m128i& mm6, __m128i& mm7)
{
  auto mm2 = mm0;
  mm0 = _mm_max_epu8(mm0, mm6);
  mm0 = _mm_min_epu8(mm0, mm7);
  mm0 = _mm_cmpeq_epi8(mm0, mm2);
  auto mm1 = mm0;
  mm0 = _mm_andnot_si128(mm0, pixel_mask);
  mm1 = _mm_and_si128(mm1, mm2);
  mm1 = _mm_or_si128(mm1, mm0);
  return mm1;
}
#endif

// check treshold on byte
// 0 in mm7
// Treshold in mm6
// result in mm1
// What it does is compute an absolute difference 
// between the current pixel (mm0) and the pixel mask (the two psubusb & por)
// Next we substract the treshold mask -> all values = 0 are good and
// must be kept, others must be replaced by the mask
AVS_FORCEINLINE __m128i check_bthreshold(__m128i orig, __m128i full_f, __m128i pixel_mask, __m128i mm6_thresh, __m128i mm7_zero)
{
  auto absdiff = _mm_subs_epu8(_mm_max_epu8(orig, pixel_mask), _mm_min_epu8(orig, pixel_mask)); // abs diff
  auto belowthresh = _mm_subs_epu8(absdiff, mm6_thresh);
  auto comp_res = _mm_cmpeq_epi8(belowthresh, mm7_zero);
  // blend
  auto mm1 = _mm_andnot_si128(comp_res, pixel_mask);
  auto mm0 = _mm_and_si128(comp_res, orig);
  return _mm_or_si128(mm0, mm1);
}


// Source is in where
// results are in mm4 and mm5
// input/output: mm4 and mm5
AVS_FORCEINLINE void compute_mul_pixel(const uint8_t *where, int mul,
  __m128i full_f, __m128i mm3_pixel_mask, __m128i mm6_thresh, __m128i mm7_zero, __m128i& mm4, __m128i& mm5)
{
  auto mm0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(where)); // movq mm0, [where]
  auto res = check_bthreshold(mm0, full_f, mm3_pixel_mask, mm6_thresh, mm7_zero); // CHECK_BTRESHOLD(full_f, mm3)
  auto mm1 = _mm_unpacklo_epi8(res, mm7_zero);
  auto mm2 = _mm_unpackhi_epi8(res, mm7_zero);
  mm1 = _mm_slli_epi16(mm1, mul);
  mm2 = _mm_slli_epi16(mm2, mul);
  mm4 = _mm_add_epi16(mm4, mm1);
  mm5 = _mm_add_epi16(mm5, mm2);
}

// Source is in where
// input/output: mm4 and mm5
AVS_FORCEINLINE void compute_pixel(const uint8_t* where,
  __m128i full_f, __m128i mm3_pixel_mask, __m128i mm6_thresh, __m128i mm7_zero, __m128i& mm4, __m128i& mm5)
{
  auto mm0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(where));
  auto res = check_bthreshold(mm0, full_f, mm3_pixel_mask, mm6_thresh, mm7_zero);
  auto mm1 = _mm_unpacklo_epi8(res, mm7_zero);
  auto mm2 = _mm_unpackhi_epi8(res, mm7_zero);
  mm4 = _mm_add_epi16(mm4, mm1);
  mm5 = _mm_add_epi16(mm5, mm2);
}
