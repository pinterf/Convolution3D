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


DWORD WINAPI SMP_NAME (void * my_data_args)
{
	__declspec(align(8)) static __int64 full_f = 0xFFFFFFFFFFFFFFFFi64;
	__declspec(align(8)) static __int64 round_32 = 0x0020002000200020i64;		// 32
	__declspec(align(8)) static __int64 round_8 = 0x0008000800080008i64;		// 8
	__declspec(align(8)) static __int64 multi_27 = 0x04BE04BE04BE04BEi64;		// = 32768/27 = 1214 in dec
	__declspec(align(8)) static __int64 round_27 = 0x001B001B001B001Bi64;		// 1B = 27
	__declspec(align(8)) static __int64 multi_11 = 0x0BA30BA30BA30BA3i64;		// = 32768/11 = 2979 in dec
	__declspec(align(8)) static __int64 round_11 = 0x000B000B000B000Bi64;		// B = 11  


	int w8;
	const unsigned char *saved_fcp, *saved_fcc, *saved_fcn, *fcp, *fcc, *fcn;
	unsigned char *saved_dest, *dest;
	int width, height, matrix, pitch_p, pitch_c, pitch_n, pitch_d;
	__int64 temp_thresh_mask, thresh_mask;
	HANDLE event;
	struct thread_data *mydata;

	mydata = (struct thread_data *) my_data_args;

	fcp = saved_fcp = mydata->saved_fcp;
	fcc = saved_fcc = mydata->saved_fcc;
	fcn = saved_fcn = mydata->saved_fcn;
	width = mydata->width;
	height = mydata->height;
	matrix = mydata->matrix;
	pitch_p = mydata->pitch_p;
	pitch_c = mydata->pitch_c;
	pitch_n = mydata->pitch_n;
	pitch_d = mydata->pitch_d;
	event = mydata->event;
	temp_thresh_mask = mydata->temp_thresh_mask;
	thresh_mask = mydata->thresh_mask;


	w8 = (width >> 3);

	dest = saved_dest = mydata->saved_dest;

	__asm
	{
		pxor mm7, mm7

		mov esi, fcc
		mov eax, fcp
		mov ebx, fcn
		mov edi, dest
		__asm V_COMPUTE (0, pitch_p, 0, pitch_c, 0, pitch_n, FirstLine121, FirstLine3D, FirstTheEnd)

		mov esi, saved_fcc
		add esi, pitch_c
		mov saved_fcc, esi
		mov eax, saved_fcp
		add eax, pitch_p
		mov saved_fcp, eax
		mov ebx, saved_fcn
		add ebx, pitch_n
		mov saved_fcn, ebx
		mov edi, saved_dest
		add edi, pitch_d
		mov saved_dest, edi
		mov edx, height
		sub edx, 2
		VLine:

		__asm V_COMPUTE (pitch_p, pitch_p, pitch_c, pitch_c, pitch_n, pitch_n, AllLine121, AllLine3D, AllTheEnd)
				
		mov esi, saved_fcc
		add esi, pitch_c
		mov saved_fcc, esi
		mov eax, saved_fcp
		add eax, pitch_p
		mov saved_fcp, eax
		mov ebx, saved_fcn
		add ebx, pitch_n
		mov saved_fcn, ebx
		mov edi, saved_dest
		add edi, pitch_d
		mov saved_dest, edi
		dec edx
		jnz VLine

		__asm V_COMPUTE (pitch_p, 0, pitch_c, 0, pitch_n, 0, LastLine121, LastLine3D, LastTheEnd)
	}

	SetEvent (event);
	return 0;
}