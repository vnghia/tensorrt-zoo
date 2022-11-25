#pragma once

#define COMPILE_TEMPLATE_BASIC_TYPES_CLASS(className) \
  COMPILE_TEMPLATE_BASIC_TYPES(className, class)
#define COMPILE_TEMPLATE_BASIC_TYPES_STRUCT(className) \
  COMPILE_TEMPLATE_BASIC_TYPES(className, struct)
#define COMPILE_TEMPLATE_BASIC_TYPES(className, classType) \
  template classType className<char>;                      \
  template classType className<signed char>;               \
  template classType className<short>;                     \
  template classType className<int>;                       \
  template classType className<long>;                      \
  template classType className<long long>;                 \
  template classType className<unsigned char>;             \
  template classType className<unsigned short>;            \
  template classType className<unsigned int>;              \
  template classType className<unsigned long>;             \
  template classType className<unsigned long long>;        \
  template classType className<float>;                     \
  template classType className<double>;                    \
  template classType className<long double>
