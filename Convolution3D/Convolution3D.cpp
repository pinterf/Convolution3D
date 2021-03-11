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

#include <windows.h>
//#define _ASSERTE(n) assert(n)
//#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "avisynth.h"
#include "c3dcommon.h"  // common define
#include "c3dyv12b.h"
#include "c3dyv12_fast.h"
#define V_COMPUTE(pl_pitch_p, nl_pitch_p, pl_pitch_c, nl_pitch_c, pl_pitch_n, nl_pitch_n,Label,Label3D,TheEnd) V_COMPUTE2(pl_pitch_p, nl_pitch_p, pl_pitch_c, nl_pitch_c, pl_pitch_n, nl_pitch_n,Label,Label3D,TheEnd,FUNC_TYPE) 
#define V_COMPUTE2(pl_pitch_p, nl_pitch_p, pl_pitch_c, nl_pitch_c, pl_pitch_n, nl_pitch_n,Label,Label3D,TheEnd,functyp) V_COMPUTE3(pl_pitch_p, nl_pitch_p, pl_pitch_c, nl_pitch_c, pl_pitch_n, nl_pitch_n,Label,Label3D,TheEnd,functyp) 
#define V_COMPUTE3(pl_pitch_p, nl_pitch_p, pl_pitch_c, nl_pitch_c, pl_pitch_n, nl_pitch_n,Label,Label3D,TheEnd,functyp) COMPUTE_##functyp##(pl_pitch_p, nl_pitch_p, pl_pitch_c, nl_pitch_c, pl_pitch_n, nl_pitch_n,Label,Label3D,TheEnd) 

#define FUNC_NAME ProcessPlane_121
#define SMP_NAME ProcessPlane_121_SMP
#define FUNC_TYPE 121
#include "c3dFunction.inc"
#include "c3dSMPFunction.inc"
#undef FUNC_NAME
#undef SMP_NAME
#undef FUNC_TYPE

#define FUNC_NAME ProcessPlane_111
#define SMP_NAME ProcessPlane_111_SMP
#define FUNC_TYPE 111
#include "c3dFunction.inc"
#include "c3dSMPFunction.inc"
#undef FUNC_NAME
#undef SMP_NAME
#undef FUNC_TYPE

#define FUNC_NAME ProcessFastPlane_121
#define SMP_NAME ProcessFastPlane_121_SMP
#define FUNC_TYPE FAST121
#include "c3dFunction.inc"
#include "c3dSMPFunction.inc"
#undef FUNC_NAME
#undef SMP_NAME
#undef FUNC_TYPE

#define FUNC_NAME ProcessFastPlane_111
#define SMP_NAME ProcessFastPlane_111_SMP
#define FUNC_TYPE FAST111
#include "c3dFunction.inc"
#include "c3dSMPFunction.inc"
#undef FUNC_NAME
#undef SMP_NAME
#undef FUNC_TYPE

//#include "c3dsmp.h"


class Convolution3D : public GenericVideoFilter
{
  int debug;
  unsigned char* y_feedback, * u_feedback, * v_feedback;
  void (*funcPtr) (const unsigned char* saved_fcp, int pitch_p,
    const unsigned char* saved_fcc, int pitch_c,
    const unsigned char* saved_fcn, int pitch_n,
    unsigned char* dest, int pitch_d,
    int width, int height,
    __int64 temp_thresh_mask,
    __int64 thresh_mask);
  DWORD(WINAPI* funcPtr_SMP) (void* my_data_args);
  BOOL smp, copyLuma, copyChroma;
  short luma_Treshold, chroma_Treshold;
  short temporal_luma_Treshold, temporal_chroma_Treshold;
  double temporal_influence;
  short luma_limit;
  short matrix;
  __int64 ythreshold_mask, ytemporal_threshold_mask;
  __int64 cthreshold_mask, ctemporal_threshold_mask;
  __int64 btreshold_mask, btemporal_treshold_mask;
  PVideoFrame fc, fp, fn; // source
  PVideoFrame final;

public:
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

  Convolution3D(PClip _child, short _matrix, short _luma_Treshold, short _chroma_Treshold,
    short _temporal_luma_Treshold, short _temporal_chroma_Treshold,
    double _temporal_influence, int _debug, IScriptEnvironment* env)
    : GenericVideoFilter(_child), matrix(_matrix), luma_Treshold(_luma_Treshold), chroma_Treshold(_chroma_Treshold),
    temporal_luma_Treshold(_temporal_luma_Treshold), temporal_chroma_Treshold(_temporal_chroma_Treshold),
    temporal_influence(_temporal_influence), debug(_debug)
  {
    SYSTEM_INFO siSysInfo;
    char dbgString[100];


    if (!vi.IsYV12())
      env->ThrowError("Convolution3D supports YV12 color format only");
    if (!(env->GetCPUFlags() & CPUF_INTEGER_SSE))
      env->ThrowError("Convolution3D needs a iSSE capable CPU");
    if (matrix < 0 || matrix > 3)
      env->ThrowError("Convolution3D : matrix parameter must be 0, 1, 2 or 3");
    if (luma_Treshold < MIN_TRESHOLD || luma_Treshold > MAX_TRESHOLD ||
      chroma_Treshold < MIN_TRESHOLD || chroma_Treshold > MAX_TRESHOLD ||
      temporal_luma_Treshold < MIN_TRESHOLD || temporal_luma_Treshold > MAX_TRESHOLD ||
      temporal_chroma_Treshold < MIN_TRESHOLD || temporal_chroma_Treshold > MAX_TRESHOLD)
      env->ThrowError("Convolution3D : all tresholds must be between 0 and 255");
    if (temporal_influence < MIN_TEMPORAL_INFLUENCE || temporal_influence > MAX_TEMPORAL_INFLUENCE)
      env->ThrowError("Convolution3D : temporal influence must be between -1 and 100");

    ythreshold_mask = (__int64)luma_Treshold << 48 | (__int64)luma_Treshold << 32 | (__int64)luma_Treshold << 16 | (__int64)luma_Treshold;
    cthreshold_mask = (__int64)chroma_Treshold << 48 | (__int64)chroma_Treshold << 32 | (__int64)chroma_Treshold << 16 | (__int64)chroma_Treshold;
    ytemporal_threshold_mask = (__int64)temporal_luma_Treshold << 48 | (__int64)temporal_luma_Treshold << 32 | (__int64)temporal_luma_Treshold << 16 | (__int64)temporal_luma_Treshold;
    ctemporal_threshold_mask = (__int64)temporal_chroma_Treshold << 48 | (__int64)temporal_chroma_Treshold << 32 | (__int64)temporal_chroma_Treshold << 16 | (__int64)temporal_chroma_Treshold;

    btreshold_mask = (__int64)chroma_Treshold << 56 | (__int64)luma_Treshold << 48 |
      (__int64)chroma_Treshold << 40 | (__int64)luma_Treshold << 32 |
      (__int64)chroma_Treshold << 24 | (__int64)luma_Treshold << 16 |
      (__int64)chroma_Treshold << 8 | (__int64)luma_Treshold;
    btemporal_treshold_mask = (__int64)temporal_chroma_Treshold << 56 | (__int64)temporal_luma_Treshold << 48 |
      (__int64)temporal_chroma_Treshold << 40 | (__int64)temporal_luma_Treshold << 32 |
      (__int64)temporal_chroma_Treshold << 24 | (__int64)temporal_luma_Treshold << 16 |
      (__int64)temporal_chroma_Treshold << 8 | (__int64)temporal_luma_Treshold;

    luma_limit = (int)(temporal_luma_Treshold * temporal_influence);
    switch (matrix)
    {
    case STANDARD_MATRIX:
      funcPtr = ProcessPlane_121;
      funcPtr_SMP = ProcessPlane_121_SMP;
      break;
    case SIMPLE_MATRIX:
      funcPtr = ProcessPlane_111;
      funcPtr_SMP = ProcessPlane_111_SMP;
      break;
    case STANDARD_FAST_MATRIX:
      funcPtr = ProcessFastPlane_121;
      funcPtr_SMP = ProcessFastPlane_121_SMP;
      break;
    case SIMPLE_FAST_MATRIX:
      funcPtr = ProcessFastPlane_111;
      funcPtr_SMP = ProcessFastPlane_111_SMP;
      break;
    }

    GetSystemInfo(&siSysInfo);
    smp = (siSysInfo.dwNumberOfProcessors > 1);
    copyChroma = (chroma_Treshold == 0) && (temporal_chroma_Treshold == 0);
    copyLuma = (luma_Treshold == 0) && (temporal_luma_Treshold == 0);
    // Force SMP
    if (debug == -2)
      smp = TRUE;

    // Disable SMP
    if (debug == -1)
      smp = FALSE;

    if (debug)
    {
      sprintf(dbgString, "Convolution3D for avisynth 2.5 v1.0.0.5\n");
      OutputDebugString(dbgString);

      sprintf(dbgString, "SMP : %d\n", smp);
      OutputDebugString(dbgString);
    }
    SetCacheHints(CACHE_WINDOW, 2);

  };

private:

};


PVideoFrame __stdcall Convolution3D::GetFrame(int n, IScriptEnvironment* env)
{
  struct thread_data d_thread_y, d_thread_u, d_thread_v;
  DWORD dwDummy;
  HANDLE ThreadIDy, ThreadIDu, ThreadIDv;
  HANDLE y_event, u_event, v_event;


  fc = child->GetFrame(n, env);
  fp = child->GetFrame(n == 0 ? 0 : n - 1, env);
  fn = child->GetFrame(n >= vi.num_frames - 1 ? vi.num_frames - 1 : n + 1, env);
  final = env->NewVideoFrame(vi);

  if (smp)
  {
    y_event = CreateEvent(NULL, 0, 0, NULL);
    u_event = CreateEvent(NULL, 0, 0, NULL);
    v_event = CreateEvent(NULL, 0, 0, NULL);

    d_thread_y.saved_fcp = fp->GetReadPtr(PLANAR_Y);
    d_thread_y.saved_fcc = fc->GetReadPtr(PLANAR_Y);
    d_thread_y.saved_fcn = fn->GetReadPtr(PLANAR_Y);
    d_thread_y.saved_dest = final->GetWritePtr(PLANAR_Y);
    d_thread_y.pitch_p = fp->GetPitch(PLANAR_Y);
    d_thread_y.pitch_c = fc->GetPitch(PLANAR_Y);
    d_thread_y.pitch_n = fn->GetPitch(PLANAR_Y);
    d_thread_y.pitch_d = final->GetPitch(PLANAR_Y);
    d_thread_y.width = fc->GetRowSize(PLANAR_Y_ALIGNED);
    d_thread_y.height = fc->GetHeight(PLANAR_Y);
    d_thread_y.matrix = matrix;
    d_thread_y.temp_thresh_mask = ytemporal_threshold_mask;
    d_thread_y.thresh_mask = ythreshold_mask;
    d_thread_y.event = y_event;

    d_thread_u.saved_fcp = fp->GetReadPtr(PLANAR_U);
    d_thread_u.saved_fcc = fc->GetReadPtr(PLANAR_U);
    d_thread_u.saved_fcn = fn->GetReadPtr(PLANAR_U);
    d_thread_u.saved_dest = final->GetWritePtr(PLANAR_U);
    d_thread_u.pitch_p = fp->GetPitch(PLANAR_U);
    d_thread_u.pitch_c = fc->GetPitch(PLANAR_U);
    d_thread_u.pitch_n = fn->GetPitch(PLANAR_U);
    d_thread_u.pitch_d = final->GetPitch(PLANAR_U);
    d_thread_u.width = fc->GetRowSize(PLANAR_U_ALIGNED);
    d_thread_u.height = fc->GetHeight(PLANAR_U);
    d_thread_u.matrix = matrix;
    d_thread_u.temp_thresh_mask = ctemporal_threshold_mask;
    d_thread_u.thresh_mask = cthreshold_mask;
    d_thread_u.event = u_event;

    d_thread_v.saved_fcp = fp->GetReadPtr(PLANAR_V);
    d_thread_v.saved_fcc = fc->GetReadPtr(PLANAR_V);
    d_thread_v.saved_fcn = fn->GetReadPtr(PLANAR_V);
    d_thread_v.saved_dest = final->GetWritePtr(PLANAR_V);
    d_thread_v.pitch_p = fp->GetPitch(PLANAR_V);
    d_thread_v.pitch_c = fc->GetPitch(PLANAR_V);
    d_thread_v.pitch_n = fn->GetPitch(PLANAR_V);
    d_thread_v.pitch_d = final->GetPitch(PLANAR_V);
    d_thread_v.width = fc->GetRowSize(PLANAR_V_ALIGNED);
    d_thread_v.height = fc->GetHeight(PLANAR_V);
    d_thread_v.matrix = matrix;
    d_thread_v.temp_thresh_mask = ctemporal_threshold_mask;
    d_thread_v.thresh_mask = cthreshold_mask;
    d_thread_v.event = v_event;


    if (copyLuma)
    {
      env->BitBlt(final->GetWritePtr(PLANAR_Y), final->GetPitch(PLANAR_Y),
        fc->GetReadPtr(PLANAR_Y), fc->GetPitch(PLANAR_Y),
        fc->GetRowSize(PLANAR_Y), fc->GetHeight(PLANAR_Y));
      SetEvent(y_event);
    }
    else
      ThreadIDy = CreateThread(NULL, 0, funcPtr_SMP, &d_thread_y, 0, &dwDummy);
    //if (ThreadIDy == NULL)
    //	env->ThrowError("Error creating thread Y");

    ThreadIDu = CreateThread(NULL, 0, funcPtr_SMP, &d_thread_u, 0, &dwDummy);
    //if (ThreadIDu == NULL)
    //	env->ThrowError("Error creating thread U");

    WaitForSingleObject(u_event, INFINITE);

    ThreadIDv = CreateThread(NULL, 0, funcPtr_SMP, &d_thread_v, 0, &dwDummy);
    //if (ThreadIDv == NULL)
    //	env->ThrowError("Error creating thread V");

    WaitForSingleObject(v_event, INFINITE);
    WaitForSingleObject(y_event, INFINITE);

    CloseHandle(y_event);
    CloseHandle(u_event);
    CloseHandle(v_event);

    CloseHandle(ThreadIDy);
    CloseHandle(ThreadIDu);
    CloseHandle(ThreadIDv);
  }
  else
  {
    if (copyLuma)
      env->BitBlt(final->GetWritePtr(PLANAR_Y), final->GetPitch(PLANAR_Y),
        fc->GetReadPtr(PLANAR_Y), fc->GetPitch(PLANAR_Y),
        fc->GetRowSize(PLANAR_Y), fc->GetHeight(PLANAR_Y));
    else
      funcPtr(fp->GetReadPtr(PLANAR_Y), fp->GetPitch(PLANAR_Y),
        fc->GetReadPtr(PLANAR_Y), fc->GetPitch(PLANAR_Y),
        fn->GetReadPtr(PLANAR_Y), fn->GetPitch(PLANAR_Y),
        final->GetWritePtr(PLANAR_Y), final->GetPitch(PLANAR_Y),
        fc->GetRowSize(PLANAR_Y_ALIGNED), fc->GetHeight(PLANAR_Y),
        ytemporal_threshold_mask,
        ythreshold_mask);

    if (copyChroma)
    {
      env->BitBlt(final->GetWritePtr(PLANAR_U), final->GetPitch(PLANAR_U),
        fc->GetReadPtr(PLANAR_U), fc->GetPitch(PLANAR_U),
        fc->GetRowSize(PLANAR_U), fc->GetHeight(PLANAR_U));
      env->BitBlt(final->GetWritePtr(PLANAR_V), final->GetPitch(PLANAR_V),
        fc->GetReadPtr(PLANAR_V), fc->GetPitch(PLANAR_V),
        fc->GetRowSize(PLANAR_V), fc->GetHeight(PLANAR_V));
    }
    else
    {
      funcPtr(fp->GetReadPtr(PLANAR_U), fp->GetPitch(PLANAR_U),
        fc->GetReadPtr(PLANAR_U), fc->GetPitch(PLANAR_U),
        fn->GetReadPtr(PLANAR_U), fn->GetPitch(PLANAR_U),
        final->GetWritePtr(PLANAR_U), final->GetPitch(PLANAR_U),
        fc->GetRowSize(PLANAR_U_ALIGNED), fc->GetHeight(PLANAR_U),
        ctemporal_threshold_mask,
        cthreshold_mask);

      funcPtr(fp->GetReadPtr(PLANAR_V), fp->GetPitch(PLANAR_V),
        fc->GetReadPtr(PLANAR_V), fc->GetPitch(PLANAR_V),
        fn->GetReadPtr(PLANAR_V), fn->GetPitch(PLANAR_V),
        final->GetWritePtr(PLANAR_V), final->GetPitch(PLANAR_V),
        fc->GetRowSize(PLANAR_V_ALIGNED), fc->GetHeight(PLANAR_V),
        ctemporal_threshold_mask,
        cthreshold_mask);
    }
  }


  __asm emms
  return final;
}


AVSValue __cdecl Create_Convolution3D(AVSValue args, void* user_data, IScriptEnvironment* env) {
  return new Convolution3D(args[0].AsClip(), args[1].AsInt(0),					// Matrix choice
    args[2].AsInt(3), args[3].AsInt(4),  // Spatial treshold
    args[4].AsInt(3), args[5].AsInt(4),  // Temporal treshold
    args[6].AsFloat(3),
    args[7].AsInt(0), env);				// debug
}

AVSValue __cdecl Create_Convolution3D_Pre(AVSValue args, void* user_data, IScriptEnvironment* env) {
  const char* myString = args[1].AsString("");
  if (!stricmp(myString, "movieHQ"))
  {
    return new Convolution3D(args[0].AsClip(), 0, 3, 4, 3, 4, 2.8, 0, env);
  }
  else if (!stricmp(myString, "movieLQ"))
  {
    return new Convolution3D(args[0].AsClip(), 0, 6, 10, 6, 8, 2.8, 0, env);
  }
  else if (!stricmp(myString, "animeHQ"))
  {
    return new Convolution3D(args[0].AsClip(), 0, 6, 12, 6, 8, 2.8, 0, env);
  }
  else if (!stricmp(myString, "animeLQ"))
  {
    return new Convolution3D(args[0].AsClip(), 1, 8, 16, 8, 8, 2.8, 0, env);
  }
  else if (!stricmp(myString, "animeBQ"))
  {
    return new Convolution3D(args[0].AsClip(), 1, 12, 22, 8, 8, 2.8, 0, env);
  }
  else if (!stricmp(myString, "vhsBQ"))
  {
    return new Convolution3D(args[0].AsClip(), 0, 16, 48, 10, 32, 4, 0, env);
  }
  else
  {
    env->ThrowError("Correct preset values are : movie[HQ, LQ], anime[HQ, LQ, BQ] or vhs[BQ]");
  }
  return new Convolution3D(args[0].AsClip(), 1, 6, 12, 6, 8, 2.8, 0, env);
}


const AVS_Linkage* AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
  AVS_linkage = vectors;

  env->AddFunction("Convolution3D", "c[matrix]i[ythresh]i[cthresh]i[t_ythresh]i[t_cthresh]i[influence]f[debug]i", Create_Convolution3D, 0);
  env->AddFunction("Convolution3D", "c[preset]s", Create_Convolution3D_Pre, 0);
  return "`Convolution3D' Denoiser";
}
