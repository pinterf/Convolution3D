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
// The value to compute must be in mm0 and the result will be in mm1
// mm7 = upper limit
// mm6 = lower limit
// pixel_mask = default value
#define CHECK_TRESHOLD(full_f,pixel_mask) __asm \
	{								\
		__asm movq mm2, mm0			\
		__asm pmaxsw mm0, mm6		\
		__asm pminsw mm0, mm7		\
		__asm pcmpeqw mm0, mm2		\
		__asm movq mm1, mm0			\
		__asm pandn mm0, pixel_mask	\
		__asm pand mm1, mm2			\
		__asm por mm1, mm0			\
	}
#endif

#if 0
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

// Check treshold on word
// FIXME PF: seems it's not word but byte!
// The value to compute must be in mm0 and the result will be in mm1
// mm7 = upper limit
// mm6 = lower limit
// pixel_mask = default value
#define CHECK_LTRESHOLD(full_f,pixel_mask) __asm \
	{								\
		__asm movq mm2, mm0			\
		__asm pmaxub mm0, mm6		\
		__asm pminub mm0, mm7		\
		__asm pcmpeqb mm0, mm2		\
		__asm movq mm1, mm0			\
		__asm pandn mm0, pixel_mask	\
		__asm pand mm1, mm2			\
		__asm por mm1, mm0			\
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
__m128i check_bthreshold(__m128i orig, __m128i full_f, __m128i pixel_mask, __m128i mm6_thresh, __m128i mm7_zero)
{
  auto absdiff = _mm_subs_epu8(_mm_max_epu8(orig, pixel_mask), _mm_min_epu8(orig, pixel_mask)); // abs diff
  auto belowthresh = _mm_subs_epu8(absdiff, mm6_thresh);
  auto comp_res = _mm_cmpeq_epi8(belowthresh, mm7_zero);
  // blend
  auto mm1 = _mm_andnot_si128(comp_res, pixel_mask);
  auto mm0 = _mm_and_si128(comp_res, orig);
  return _mm_or_si128(mm0, mm1);
}


// check treshold on byte
// 0 in mm7
// Treshold in mm6
// result in mm1
// What it does is compute an absolute difference 
// between the current pixel (mm0) and the pixel mask (the two psubusb & por)
// Next we substract the treshold mask -> all values = 0 are good and
// must be kept, others must be replaced by the mask
#define CHECK_BTRESHOLD(full_f, pixel_mask) __asm \
	{													\
		__asm movq mm1, pixel_mask						\
		__asm movq mm2, mm0								\
		__asm pmaxub mm0, pixel_mask					\
		__asm pminub mm1, mm2							\
		__asm psubusb mm0, mm1							\
		__asm psubusb mm0, mm6							\
		__asm pcmpeqb mm0, mm7							\
		__asm movq mm1, mm0								\
		__asm pand mm0, mm2								\
		__asm pandn mm1, pixel_mask						\
		__asm por mm1, mm0								\
	}


// Source is in where
// results are in mm4 and mm5
// input/output: mm4 and mm5
void compute_mul_pixel(const uint8_t *where, int mul, 
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
// result in mm1
#define COMPUTE_MUL_PIXEL(where, mul) __asm	\
{												\
	__asm movq mm0, [where]						\
	__asm CHECK_BTRESHOLD(full_f, mm3)			\
	__asm movq mm2, mm1							\
	__asm punpcklbw mm1, mm7					\
	__asm punpckhbw mm2, mm7					\
	__asm psllw mm1, mul						\
	__asm psllw mm2, mul						\
	__asm paddw mm4, mm1						\
	__asm paddw mm5, mm2						\
}



// Source is in where
// results are in mm4 and mm5
// input/output: mm4 and mm5
void compute_pixel(const uint8_t* where,
  __m128i full_f, __m128i mm3_pixel_mask, __m128i mm6_thresh, __m128i mm7_zero, __m128i& mm4, __m128i& mm5)
{
  auto mm0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(where));
  auto res = check_bthreshold(mm0, full_f, mm3_pixel_mask, mm6_thresh, mm7_zero);
  auto mm1 = _mm_unpacklo_epi8(res, mm7_zero);
  auto mm2 = _mm_unpackhi_epi8(res, mm7_zero);
  mm4 = _mm_add_epi16(mm4, mm1);
  mm5 = _mm_add_epi16(mm5, mm2);
}

// Source is in where

// input/output: mm4 and mm5
#define COMPUTE_PIXEL(where) __asm				\
{												\
	__asm movq mm0, [where]						\
	__asm CHECK_BTRESHOLD(full_f, mm3)			\
	__asm movq mm2, mm1							\
	__asm punpcklbw mm1, mm7					\
	__asm punpckhbw mm2, mm7					\
	__asm paddw mm4, mm1						\
	__asm paddw mm5, mm2						\
}



struct thread_data
{
	const unsigned char *saved_fcp;
	const unsigned char *saved_fcc;
	const unsigned char *saved_fcn;
	unsigned char *saved_dest;
	int pitch_p;
	int pitch_c;
	int pitch_n;
	int pitch_d;
	int width;
	int height;
	int matrix;
	__int64 temp_thresh_mask;
	__int64 thresh_mask; 
	HANDLE event;
};



/*
mov mm2, mm0
pmaxsw mm0, Down_Limit
pminsw mm0, Up_Limit
pcmpeqw mm0, mm2		// mm0 = FF where value should be kept
movq mm1, mm0
pxor mm0, full_f
pand mm0, mm2
pand mm1, pixel_mask
por mm1, mm0

  
	
movq mm1, pixel_mask  // copy center value
movq mm2, mm0         // computed_value
pmaxsw mm0, mm1
pminsw mm1, mm2
psubw mm0, mm1
psubusw mm0, Threshold_Mask
pcmpeqw mm0, Full_zeros
movq mm1, mm0
pxor mm0, full_f
pand mm0, mm2
pand mm1, pixel_mask
por mm1, mm0


movq mm0, thisval   ; 2 cycles
movq mm1, thatval   ; 2 cycles
movq mm2, mm0       ; 0 cycles (paired)
pminub mm0,mm1      ; 2 cycles - mm0 = min
pmaxub mm1,mm2      ; 0 cycles - mm1 = max
psubusb mm1,mm0     ; 2 cycles
                    

		__asm movq mm2, mm0			\
		__asm pmaxub mm0, mm6		\
		__asm pminub mm0, mm7		\
		__asm pcmpeqw mm0, mm1		\
		__asm movq mm1, mm0			\
		__asm pandn mm0, pixel_mask	\
		__asm pand mm1, mm2			\
		__asm por mm1, mm0			\

	V_COMPUTE (0, pitch_p, 0, pitch_c, 0, pitch_n, FirstLine, FirstLine3D, FirstTheEnd)

	saved_fcp += pitch_p;
	saved_fcc += pitch_c;
	saved_fcn += pitch_n;

	saved_dest += pitch_d;

	for (y = 1; y < height-1; y++)
	{
		fcp = saved_fcp;
		fcc = saved_fcc;
		fcn = saved_fcn;

		dest = saved_dest;
		V_COMPUTE (pitch_p, pitch_p, pitch_c, pitch_c, pitch_n, pitch_n, AllLine, AllLine3D, AllTheEnd)
			
		saved_fcp += pitch_p;
		saved_fcc += pitch_c;
		saved_fcn += pitch_n;

		saved_dest += pitch_d;
	}

	fcp = saved_fcp;
	fcc = saved_fcc;
	fcn = saved_fcn;

	dest = saved_dest;

	V_COMPUTE (pitch_p, 0, pitch_c, 0, pitch_n, 0, LastLine, LastLine3D, LastTheEnd)



*/
