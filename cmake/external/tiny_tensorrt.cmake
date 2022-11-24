if(TARGET tinytrt)
  return()
endif()

set(TINY_TENSORRT_GIT_TAG 7ac9c6c6863ca3435a3407a6241f276ad3c49672)
include(FetchContent)
FetchContent_Declare(
  tiny-tensorrt
  GIT_REPOSITORY https://github.com/zerollzeng/tiny-tensorrt.git
  GIT_TAG "${TINY_TENSORRT_GIT_TAG}"
)
FetchContent_MakeAvailable(tiny-tensorrt)
