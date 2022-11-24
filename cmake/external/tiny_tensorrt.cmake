if(TARGET tinytrt)
  return()
endif()

set(TINY_TENSORRT_GIT_TAG 3ebd2b5eb81436dd0790dae6b1b661c13bb260a0)
include(FetchContent)
FetchContent_Declare(
  tiny-tensorrt
  GIT_REPOSITORY https://github.com/vnghia/tiny-tensorrt.git
  GIT_TAG "${TINY_TENSORRT_GIT_TAG}"
)
FetchContent_MakeAvailable(tiny-tensorrt)
