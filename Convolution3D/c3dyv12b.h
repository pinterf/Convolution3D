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


// nl_pitch_c, pl_pitch_c : next line pitch_c, previous line pitch_c
//
//
//		__asm pxor mm7, mm7							\
//		__asm mov esi, fcc							\
//		__asm mov eax, fcp							\
//		__asm mov ebx, fcn							\
//		__asm mov edi, dest							\
//
//
#define COMPUTE_121(pl_pitch_p, nl_pitch_p, pl_pitch_c, nl_pitch_c, pl_pitch_n, nl_pitch_n,Label121,Label3D,TheEnd) __asm \
	{												\
		__asm mov ecx, w8							\
		__asm Label121:								\
		__asm movq mm4, [esi]						\
		__asm movq mm3, mm4							\
		__asm movq mm5, mm4							\
													\
		__asm movq mm6, thresh_mask					\
		__asm punpcklbw mm4, mm7					\
		__asm punpckhbw mm5, mm7					\
		__asm psllw mm4, 2							\
		__asm psllw mm5, 2							\
													\
		__asm COMPUTE_MUL_PIXEL (esi-1,1)			\
		__asm COMPUTE_MUL_PIXEL (esi+1,1)			\
		__asm sub esi, pl_pitch_c					\
		__asm COMPUTE_PIXEL (esi-1)					\
		__asm COMPUTE_MUL_PIXEL (esi,1)				\
		__asm COMPUTE_PIXEL (esi+1)					\
		__asm add esi, pl_pitch_c					\
		__asm add esi, nl_pitch_c					\
		__asm COMPUTE_PIXEL (esi-1)					\
		__asm COMPUTE_MUL_PIXEL (esi,1)				\
		__asm COMPUTE_PIXEL (esi+1)					\
		__asm sub esi, nl_pitch_c					\
		__asm add esi, 8							\
													\
		__asm movq mm6, temp_thresh_mask			\
		__asm psllw mm4, 1							\
		__asm psllw mm5, 1							\
													\
		__asm sub eax, pl_pitch_p					\
		__asm COMPUTE_PIXEL (eax-1)					\
		__asm COMPUTE_MUL_PIXEL (eax,1)				\
		__asm COMPUTE_PIXEL (eax+1)					\
		__asm add eax, pl_pitch_p					\
		__asm COMPUTE_MUL_PIXEL (eax-1,1)			\
		__asm COMPUTE_MUL_PIXEL (eax,2)				\
		__asm COMPUTE_MUL_PIXEL (eax+1,1)			\
		__asm add eax, nl_pitch_p					\
		__asm COMPUTE_PIXEL (eax-1)					\
		__asm COMPUTE_MUL_PIXEL (eax,1)				\
		__asm COMPUTE_PIXEL (eax+1)					\
		__asm sub eax, nl_pitch_p					\
													\
		__asm add eax, 8							\
													\
		__asm sub ebx, pl_pitch_n					\
		__asm COMPUTE_PIXEL (ebx-1)					\
		__asm COMPUTE_MUL_PIXEL (ebx,1)				\
		__asm COMPUTE_PIXEL (ebx+1)					\
		__asm add ebx, pl_pitch_n					\
		__asm COMPUTE_MUL_PIXEL (ebx-1,1)			\
		__asm COMPUTE_MUL_PIXEL (ebx,2)				\
		__asm COMPUTE_MUL_PIXEL (ebx+1,1)			\
		__asm add ebx, nl_pitch_n					\
		__asm COMPUTE_PIXEL (ebx-1)					\
		__asm COMPUTE_MUL_PIXEL (ebx,1)				\
		__asm COMPUTE_PIXEL (ebx+1)					\
		__asm sub ebx, nl_pitch_n					\
		__asm add ebx, 8							\
													\
		__asm paddw mm4, round_32					\
		__asm paddw mm5, round_32					\
		__asm psrlw mm4, 6							\
		__asm psrlw mm5, 6							\
		__asm packuswb mm4, mm5						\
		__asm movntq [edi], mm4						\
		__asm add edi, 8							\
		__asm dec ecx								\
		__asm jnz Label121							\
	}


#define COMPUTE_111(pl_pitch_p, nl_pitch_p, pl_pitch_c, nl_pitch_c, pl_pitch_n, nl_pitch_n, Label111,Label3D,TheEnd) __asm \
	{												\
		__asm mov ecx, w8							\
		__asm Label111:								\
		__asm movq mm4, [esi]						\
		__asm movq mm3, mm4							\
		__asm movq mm5, mm4							\
													\
		__asm movq mm6, thresh_mask					\
		__asm punpcklbw mm4, mm7					\
		__asm punpckhbw mm5, mm7					\
													\
		__asm COMPUTE_PIXEL (esi-1)					\
		__asm COMPUTE_PIXEL (esi+1)					\
		__asm sub esi, pl_pitch_c					\
		__asm COMPUTE_PIXEL (esi-1)					\
		__asm COMPUTE_PIXEL (esi)					\
		__asm COMPUTE_PIXEL (esi+1)					\
		__asm add esi, pl_pitch_c					\
		__asm add esi, nl_pitch_c					\
		__asm COMPUTE_PIXEL (esi-1)					\
		__asm COMPUTE_PIXEL (esi)					\
		__asm COMPUTE_PIXEL (esi+1)					\
		__asm movq mm6, temp_thresh_mask			\
		__asm sub esi, nl_pitch_c					\
		__asm add esi, 8							\
													\
		__asm sub eax, pl_pitch_p					\
		__asm COMPUTE_PIXEL (eax-1)					\
		__asm COMPUTE_PIXEL (eax)					\
		__asm COMPUTE_PIXEL (eax+1)					\
		__asm add eax, pl_pitch_p					\
		__asm COMPUTE_PIXEL (eax-1)					\
		__asm COMPUTE_PIXEL (eax)					\
		__asm COMPUTE_PIXEL (eax+1)					\
		__asm add eax, nl_pitch_p					\
		__asm COMPUTE_PIXEL (eax-1)					\
		__asm COMPUTE_PIXEL (eax)					\
		__asm COMPUTE_PIXEL (eax+1)					\
		__asm sub eax, nl_pitch_p					\
													\
		__asm add eax, 8							\
													\
		__asm sub ebx, pl_pitch_n					\
		__asm COMPUTE_PIXEL (ebx-1)					\
		__asm COMPUTE_PIXEL (ebx)					\
		__asm COMPUTE_PIXEL (ebx+1)					\
		__asm add ebx, pl_pitch_n					\
		__asm COMPUTE_PIXEL (ebx-1)					\
		__asm COMPUTE_PIXEL (ebx)					\
		__asm COMPUTE_PIXEL (ebx+1)					\
		__asm add ebx, nl_pitch_n					\
		__asm COMPUTE_PIXEL (ebx-1)					\
		__asm COMPUTE_PIXEL (ebx)					\
		__asm COMPUTE_PIXEL (ebx+1)					\
		__asm sub ebx, nl_pitch_n					\
		__asm add ebx, 8							\
													\
		__asm psllw mm4, 1							\
		__asm psllw mm5, 1							\
		__asm paddw mm4, round_27					\
		__asm paddw mm5, round_27					\
		__asm pmulhuw mm4, multi_27					\
		__asm pmulhuw mm5, multi_27					\
		__asm packuswb mm4, mm5						\
		__asm movntq [edi], mm4						\
		__asm add edi, 8							\
		__asm dec ecx								\
		__asm jnz Label111							\
	}
