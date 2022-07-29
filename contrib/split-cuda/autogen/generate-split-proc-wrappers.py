#!/usr/bin/python

'''
/****************************************************************************
 *  Copyright (C) 2019-2020 by Twinkle Jain, Rohan garg, and Gene Cooperman *
 *  jain.t@husky.neu.edu, rohgarg@ccs.neu.edu, gene@ccs.neu.edu             *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License along with DMTCP:dmtcp/src.  If not, see                        *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/
'''

# To try this out, do:
#   python <THIS_FILE> generate-example.txt

import sys
import subprocess
import os
import re

if len(sys.argv) != 3:
  print("***  Usage: " + sys.argv[0] + " <SPLIT_PROCESS.decl>")
  print("***         " + sys.argv[0] + " -  # Read declaratsion from stdin")
  print("***    " + "<SPLIT_PROCESS.decl> has lines like: void foo(int x);")
  sys.exit(1)

# change the file name here!
header_file_name = "cuda_autogen_wrappers.h"
lower_half_header_name = "lower_half_cuda_if.h"
stub_lib_name = "cuda_stub.cpp"
mode = sys.argv[2]

buffer_list=list()
string_list=list()
if sys.argv[-1] == "-":
  declarations_file = sys.stdin
  header_file = sys.stdout
else:
  declarations_file = open(sys.argv[1])
  # write declaration of wrappers in a separate file
  header_file = open("autogen/"+header_file_name, 'w')
  lower_half_header = open("autogen/"+lower_half_header_name, 'w')
  stub_lib = open("cuda_stub_lib/"+stub_lib_name, 'w')

declarations = declarations_file.read().split(';')[:-1]  # Each decl ends in ';'
declarations_file.close()

header_file.write("#ifndef "+ header_file_name.replace('.', '_').upper() + "\n")
header_file.write("#define "+ header_file_name.replace('.','_').upper() + "\n\n\n")
header_file.write("#include <cuda_runtime_api.h>\n")
header_file.write("#include <cublas_v2.h>\n")
header_file.write("#include <cusparse.h>\n")
header_file.write("#include <cusolverDn.h>\n")

lower_half_header.write("#ifndef "+ lower_half_header_name.replace('.', '_').upper() + "\n")
lower_half_header.write("#define "+ lower_half_header_name.replace('.','_').upper() + "\n\n\n")

stub_lib.write("#include <assert.h>\n")
stub_lib.write("#include <cuda.h>\n")
stub_lib.write("#include <cuda_runtime.h>\n")
stub_lib.write("#include <cuda_runtime_api.h>\n")
stub_lib.write("#include <cublas_v2.h>\n")
stub_lib.write("#include <cusolverDn.h>\n")
stub_lib.write("#include <cusparse.h>\n\n")
# =============================================================

def isDefaultStreamEnabled():
  if mode == "--stream":
    return True
  return False

def makePTDS(fnc):
  return fnc+"_ptds"

def makePTSZ(fnc):
  return fnc+"_ptsz"


def isPTDS(fnc):
  if fnc in ["cudaMemcpy", "cudaMemcpyToSymbol", "cudaMemcpyFromSymbol", \
            "cudaMemcpy2D", "cudaMemcpyToArray", "cudaMemcpy2DToArray", \
            "cudaMemcpyFromArray", "cudaMemcpy2DFromArray", \
            "cudaMemcpyArrayToArray", "cudaMemcpy2DArrayToArray", \
            "cudaMemcpy3D", "cudaMemcpy3DPeer", "cudaMemset", \
            "cudaMemset2D", "cudaMemset3D"]:
    return True
  return False

def isPTSZ(fnc):
  if fnc in ["cudaGraphLaunch", "cudaStreamBeginCapture",
             "cudaStreamEndCapture", "cudaStreamIsCapturing",
             "cudaMemcpyAsync", "cudaMemcpyToSymbolAsync",
             "cudaMemcpyFromSymbolAsync", "cudaMemcpy2DAsync",
             "cudaMemcpyToArrayAsync", "cudaMemcpy2DToArrayAsync",
             "cudaMemcpyFromArrayAsync", "cudaMemcpy2DFromArrayAsync",
             "cudaMemcpy3DAsync", "cudaMemcpy3DPeerAsync",
             "cudaMemsetAsync", "cudaMemset2DAsync",
             "cudaMemset3DAsync", "cudaStreamQuery",
             "cudaStreamGetFlags", "cudaStreamGetPriority",
             "cudaEventRecord", "cudaStreamWaitEvent",
             "cudaStreamAddCallback", "cudaStreamAttachMemAsync",
             "cudaStreamSynchronize", "cudaLaunch",
             "cudaLaunchKernel", "cudaLaunchHostFunc",
             "cudaMemPrefetchAsync", "cudaLaunchCooperativeKernel",
             "cudaSignalExternalSemaphoresAsync",
             "cudaWaitExternalSemaphoresAsync"]:
    return True
  return False

def strip_fnc(fnc):
  fnc_stripped = fnc
  if fnc.endswith('_ptsz') or fnc.endswith('_ptds'):
    fnc_stripped = fnc[:-5]
  return fnc

def abort_decl(decl, comment):
  print("*** Can't parse:  " + decl + " (" + comment + ")")
  sys.exit(1)

def get_var(arg):
  global var_idx
  words = re.split("[^a-zA-Z0-9_]+", arg.strip())
  if not words:
    abort_decl(arg, "arguments of a function declaration")
  var = words[-1] or words[-2]  # args[-1] might be empty string: int foo(int *)
  keyword = (len(words) >= 3 and not words[-1] and words[-3] or
             len(words) == 2 and words[-2])
  # if this is only a type, no var
  if (not re.match("[a-zA-Z0-9_]", var[-1]) or
      keyword in ["struct", "enum", "union"] or
      ((words[-1] or words[-2])
       in ["int", "unsigned", "signed", "float", "double", "char"])):
    var = "var" + str(var_idx)
  var_idx += 1 # increment varX for each arg position
  return var

def add_anonymous_vars_to_decl(decl, args, arg_vars):
  raw_args = []
  for (arg, var) in zip(args.split(','), arg_vars):
    if not re.match(r"\b" + var + r"\b", arg):  # if var not user-named variable
      assert re.match(r"\bvar[1-9]\b", var)
      arg += " " + var # then var must be a missing variable in decl; add it
    raw_args += [arg]
  return decl.split('(')[0] + "(" + ','.join(raw_args) + ")"

def handleComment(decl_oneline):
  if decl_oneline.endswith("*/"):
    print(decl_oneline.rstrip(';'))
  else:
    abort_decl(decl_oneline, "Comment not closed; missing ending '*/'")

def emit_wrapper(decl, ret_type, fnc, args, arg_vars, logging):
  fat = False;  
  unreg= False;  
  if re.match(r"var[1-9]\b", ' '.join(arg_vars)):
    decl = add_anonymous_vars_to_decl(decl, args, arg_vars);
  # if arg_vars contains "varX", then "var2" needs to be inserted before
  # the second comma (or before the trailing ')' for 2-arg fnc)
  header_file.write('extern "C" ' + decl + " __attribute__((weak));\n")
  header_file.write("#define " + fnc + "(" + ", ".join(arg_vars) + ") (" + fnc \
        + " ? " + fnc + "(" + ", ".join(arg_vars) + ") : 0)\n\n")
  stub_lib.write('extern "C" '+ decl+" {\n  assert(0);\n")
  buffer_list.append("MACRO("+strip_fnc(fnc)+") ,"+"\\"+"\n")
  string_list.append('  "' + fnc + '",\n')
  print ("\n#undef "+ fnc)
  print('extern "C" ' + decl + " {")
  print("  typedef " + ret_type +
        " (*"+ fnc +"_t)(" + decl.split('(', 1)[1] + ";")

  if ret_type in 'cudaError_t':
    init_val = '= cudaSuccess;'
  elif '*' in ret_type:
    init_val = '= NULL;'
  elif ret_type in ["cudaChannelFormatDesc", "cublasStatus_t", "cusparseStatus_t", \
                    "cusparseMatrixType_t", "cusparseFillMode_t", "cusparseDiagType_t", \
                    "cusparseIndexBase_t", "cusolverStatus_t"]:
    init_val = ';'
  elif ret_type in 'CUresult':
    init_val = '= CUDA_SUCCESS;'
  else:
    init_val = '= 0;'
  if ret_type != "void":
    print("  " + ret_type + ' ret_val ' + init_val)
  #  print("  fnc_ptr_t fnc_ptr = get_fnc_ptr(\"" + fnc + "\");")
  if fnc == "__cudaRegisterFatBinary":
    fat= True;
  elif fnc == "__cudaUnregisterFatBinary":
    print("  fatHandle_t fat= NULL;")
    print(" // fat = (fatHandle_t)lhInfo.new_getFatCubinHandle;")
    print("  fat = (fatHandle_t)lhInfo.getFatCubinHandle;")
    unreg= True;
  print("  DMTCP_PLUGIN_DISABLE_CKPT();")
  if (unreg):
      print("  if (fat() == NULL) {")
      print("   fatCubinHandle = global_fatCubinHandle;")
      print("  } else {")
      print("   fatCubinHandle = fat();")
      print("  }");
      print("  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);")
      print("  REAL_FNC(" + strip_fnc(fnc) + ")(" + " fatCubinHandle " + ");") 
      print("  RETURN_TO_UPPER_HALF();")
      if(logging):
        print("/* Insert logging code here */")
        print("  logAPI(Cuda_Fnc_"+ fnc + ", " +  "fatCubinHandle" + ");")
  else:
   print("  JUMP_TO_LOWER_HALF(lhInfo.lhFsAddr);")
   if ret_type != "void":
    print("  ret_val = "
          "REAL_FNC(" + strip_fnc(fnc) + ")(" + ", ".join(arg_vars) + ");")
    stub_lib.write("  " + ret_type + ' ret_val ' + init_val + "\n")
   else:
    print("  REAL_FNC(" + strip_fnc(fnc) + ")(" + ", ".join(arg_vars) + ");")

   print("  RETURN_TO_UPPER_HALF();")

   if (logging):
    print("/* Insert logging code here */")
    if ret_type != "void":
      print("  logAPI(Cuda_Fnc_"+ fnc + ", " +  ", ".join(arg_vars) + ", ret_val);")
    else:
      print("  logAPI(Cuda_Fnc_"+ fnc + ", " +  ", ".join(arg_vars) + ");")
  if (fat):
    print("  global_fatCubinHandle = ret_val;")
    fat= False;
  print("  DMTCP_PLUGIN_ENABLE_CKPT();")

# return code
  if ret_type != "void":
    print("  return ret_val;")
    stub_lib.write("  return ret_val;\n")

  stub_lib.write("}\n\n")
  print("}")

# initial value
logging = False
skip_decl = False
cuda_ver_str = re.findall(r"release \d+",
                          subprocess.check_output(["nvcc", "--version"])
                          .decode('utf-8'))[0]
cuda_ver = int(cuda_ver_str.split(" ")[1])

for decl in declarations:
  # check for header file
  decl_oneline = re.sub('\n *', ' ', decl).strip()

  # skip the current declaration if the skip_decl flag is set
  if skip_decl:
    if decl_oneline.startswith("@LogReplay"):
      continue
    skip_decl = False # unset
    logging = False
    continue

  if decl_oneline.startswith("#"):
    print(decl_oneline.rstrip(';'))
    continue

  if decl_oneline.startswith("/*"):
    handleComment(decl_oneline)
    continue

  if decl_oneline.startswith("@LogReplay"):
    logging = True
    continue

  if decl_oneline.startswith("@VERSION"):
    decl_ver = int(decl_oneline.split("_")[1])
    if decl_ver != cuda_ver:
      skip_decl = True
    continue

  if decl.rstrip()[-1] != ')':
    abort_decl(decl, "missing final ')'")
  if '(' not in decl:
    abort_decl(decl, "missing '('")
  (ret_type_and_fnc, args) = decl_oneline[:-1].split('(', 1)

  var_idx = 1
  fnc = get_var(ret_type_and_fnc)
  ret_type = ret_type_and_fnc.rstrip().rsplit(fnc, 1)[0].strip()

  if isDefaultStreamEnabled():
    if isPTSZ(fnc):
      fnc = makePTSZ(fnc)
    elif isPTSZ(fnc):
      fnc = makePTDS(fnc)

  var_idx = 1
  if args.strip(): # if one or more arguments
    arg_vars = [get_var(arg) for arg in args.split(',')]
  else:  # else this is a function of zero arguments
    arg_vars = []

  emit_wrapper(decl_oneline, ret_type, fnc, args, arg_vars, logging)
  print("")  # emit a newline
  logging = False # reset

b="".join(buffer_list)
b=b[:-2]
lower_half_header.write("#define FOREACH_FNC(MACRO) \\"+"\n"+b+"\n")
stub_code = '#define GENERATE_ENUM(ENUM) Cuda_Fnc_##ENUM\n\n' \
          + '#define GENERATE_FNC_PTR(FNC) ((void*)&FNC)\n\n' \
          + 'typedef enum __Cuda_Fncs {\n' \
          + '  Cuda_Fnc_NULL,\n' \
          + '  FOREACH_FNC(GENERATE_ENUM)\n' \
          + '  Cuda_Fnc_Invalid,\n' \
          + '} Cuda_Fncs_t;\n\n'

strlist = "".join(string_list)
string_code = "static const char *cuda_Fnc_to_str[]  __attribute__((used)) =\n{\n" \
            + '  "Cuda_Fnc_NULL", \n' \
            + strlist \
            + '  "Cuda_Fnc_Invalid"\n};\n'


lower_half_header.write(stub_code)
lower_half_header.write(string_code)

header_file.write("#endif // "+ header_file_name.replace('.', '_').upper())
lower_half_header.write("#endif // "+ lower_half_header_name.replace('.', '_').upper())
header_file.close()
lower_half_header.close()
stub_lib.close()


'''
@LogReplay;
void**
__cudaRegisterFatBinary(void *fatCubin);

@LogReplay;
void
__cudaUnregisterFatBinary(void **fatCubinHandle);
'''
