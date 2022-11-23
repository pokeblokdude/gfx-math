#ifndef _GFX_MATH_H
#define _GFX_MATH_H

// CREDIT: constant definitions taken from the built-in <math.h> library
#define M_E		2.7182818284590452354
#define M_LOG2E		1.4426950408889634074	/* log_2 e */
#define M_LOG10E	0.43429448190325182765	/* log_10 e */
#define M_LN2		0.69314718055994530942	/* log_e 2 */
#define M_LN10		2.30258509299404568402	/* log_e 10 */
#define M_PI		3.14159265358979323846	/* pi */
#define M_PI_2		1.57079632679489661923	/* pi/2 */
#define M_PI_4		0.78539816339744830962	/* pi/4 */
#define M_1_PI		0.31830988618379067154	/* 1/pi */
#define M_2_PI		0.63661977236758134308	/* 2/pi */
#define M_2_SQRTPI	1.12837916709551257390	/* 2/sqrt(pi) */
#define M_SQRT2		1.41421356237309504880	/* sqrt(2) */
#define M_SQRT1_2	0.70710678118654752440	/* 1/sqrt(2) */

#define M_4_OVER_PI 1.27323954474
#define M_THREE_HALF_PI 4.71238898038

// initial list of functions taken from HLSL docs
// https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-intrinsic-functions

// CREDIT FOR OPERATOR OVERRIDES: https://en.cppreference.com/w/cpp/language/operators
struct vec2 {
  float x;
  float y;

  // override addition
  vec2& operator+=(const vec2& rhs) {
    this->x += rhs.x;
    this->y += rhs.y;
    return *this;
  }
  friend vec2 operator+(vec2 lhs, const vec2& rhs) {
    lhs += rhs;
    return lhs;
  }
  // override subtraction
  vec2& operator-=(const vec2& rhs) {
    this->x -= rhs.x;
    this->y -= rhs.y;
    return *this;
  }
  friend vec2 operator-(vec2 lhs, const vec2& rhs) {
    lhs -= rhs;
    return lhs;
  }
  // override v/v multiplication
  vec2& operator*=(const vec2& rhs) {
    this->x *= rhs.x;
    this->y *= rhs.y;
    return *this;
  }
  friend vec2 operator*(vec2 lhs, const vec2& rhs) {
    lhs *= rhs;
    return lhs;
  }
  // override v/f multiplication
  vec2& operator*=(const float& rhs) {
    this->x *= rhs;
    this->y *= rhs;
    return *this;
  }
  friend vec2 operator*(vec2 lhs, const float& rhs) {
    lhs *= rhs;
    return lhs;
  }
  // f/v mult
  friend vec2 operator*(const float& lhs, vec2 rhs) {
    rhs *= lhs;
    rhs *= lhs;
    return rhs;
  }
  // override negation
  vec2& operator-() const {
    vec2 v;
    v.x = -this->x;
    v.y = -this->y;
    return v;
  }
};
struct vec3 {
  float x;
  float y;
  float z;

  vec3& operator+=(const vec3& rhs) {
    this->x += rhs.x;
    this->y += rhs.y;
    this->z += rhs.z;
    return *this;
  }
  friend vec3 operator+(vec3 lhs, const vec3& rhs) {
    lhs += rhs;
    return lhs;
  }
  vec3& operator-=(const vec3& rhs) {
    this->x -= rhs.x;
    this->y -= rhs.y;
    this->z -= rhs.z;
    return *this;
  }
  friend vec3 operator-(vec3 lhs, const vec3& rhs) {
    lhs -= rhs;
    return lhs;
  }
  vec3& operator*=(const vec3& rhs) {
    this->x *= rhs.x;
    this->y *= rhs.y;
    this->z *= rhs.z;
    return *this;
  }
  friend vec3 operator*(vec3 lhs, const vec3& rhs) {
    lhs *= rhs;
    return lhs;
  }
  vec3& operator*=(const float& rhs) {
    this->x *= rhs;
    this->y *= rhs;
    this->z *= rhs;
    return *this;
  }
  friend vec3 operator*(vec3 lhs, const float& rhs) {
    lhs *= rhs;
    return lhs;
  }
  friend vec3 operator*(const float& lhs, vec3 rhs) {
    rhs *= lhs;
    rhs *= lhs;
    return rhs;
  }
  vec3& operator-() const {
    vec3 v;
    v.x = -this->x;
    v.y = -this->y;
    v.z = -this->z;
    return v;
  }
};
struct vec4 {
  float x;
  float y;
  float z;
  float w;

  vec4& operator+=(const vec4& rhs) {
    this->x += rhs.x;
    this->y += rhs.y;
    this->z += rhs.z;
    this->w += rhs.w;
    return *this;
  }
  friend vec4 operator+(vec4 lhs, const vec4& rhs) {
    lhs += rhs;
    return lhs;
  }
  vec4& operator-=(const vec4& rhs) {
    this->x -= rhs.x;
    this->y -= rhs.y;
    this->z -= rhs.z;
    this->w -= rhs.w;
    return *this;
  }
  friend vec4 operator-(vec4 lhs, const vec4& rhs) {
    lhs -= rhs;
    return lhs;
  }
  vec4& operator*=(const vec4& rhs) {
    this->x *= rhs.x;
    this->y *= rhs.y;
    this->z *= rhs.z;
    this->w *= rhs.w;
    return *this;
  }
  friend vec4 operator*(vec4 lhs, const vec4& rhs) {
    lhs *= rhs;
    return lhs;
  }
  vec4& operator*=(const float& rhs) {
    this->x *= rhs;
    this->y *= rhs;
    this->z *= rhs;
    this->w *= rhs;
    return *this;
  }
  friend vec4 operator*(vec4 lhs, const float& rhs) {
    lhs *= rhs;
    return lhs;
  }
  friend vec4 operator*(const float& lhs, vec4 rhs) {
    rhs *= lhs;
    rhs *= lhs;
    return rhs;
  }
  vec4& operator-() const {
    vec4 v;
    v.x = -this->x;
    v.y = -this->y;
    v.z = -this->z;
    v.w = -this->w;
    return v;
  }
};

typedef float mat2x2[2][2];
typedef float mat3x3[3][3];
typedef float mat4x4[4][4];

// absolute value of each component of x
float abs(float x);
vec2 abs(vec2 v);
vec3 abs(vec3 v);
vec4 abs(vec4 v);
float* abs(mat2x2 m);
float* abs(mat3x3 m);
float* abs(mat4x4 m);

// returns true if all components are non-zero
bool all(float x);
bool all(vec2 v);
bool all(vec3 v);
bool all(vec4 v);
bool all(mat2x2 m);
bool all(mat3x3 m);
bool all(mat4x4 m);

// arctangent of two values
float atan2(float y, float x);
vec2 atan2(vec2 v1, vec2 v0);
vec3 atan2(vec3 v1, vec3 v0);
vec4 atan2(vec4 v1, vec4 v0);
void atan2(mat2x2 m1, mat2x2 m0);
void atan2(mat3x3 m1, mat3x3 m0);
void atan2(mat4x4 m1, mat4x4 m0);

// ceil

// clamp value between min and max
float clamp(float x, float min, float max);

// clamp component-wise between 0 and 1
float clamp01(float x);
vec2 clamp01(vec2 v);
vec3 clamp01(vec3 v);
vec4 clamp01(vec4 v);

// cosine of each component
float cos(float x);

// cross product of two vectors
vec3 cross(vec3 a, vec3 b);

// convert from radians to degrees
float degrees(float r);

// return determinant of square matrix
float determinant(mat2x2 m);
float determinant(mat3x3 m);
float determinant(mat4x4 m);

// return distance between two points
float distance(vec2 v0, vec2 v1);
float distance(vec3 v0, vec3 v1);
float distance(vec4 v0, vec4 v1);

// dot product of two vectors
float dot(vec2 a, vec2 b);
float dot(vec3 a, vec3 b);
float dot(vec4 a, vec4 b);

// base-e exponential
float exp(float x);

// base-2 exponential
float exp2(float x);

// floating-point remainder of x/y
float fmod(float x, float y);

// returns the fractional part of x
float frac(float x);

// return length of vector
float length(vec2 v);
float length(vec3 v);
float length(vec4 v);

// component-wise linear interpolation
float lerp(float v0, float v1, float t);
vec2 lerp(vec2 v0, vec2 v1, float t);
vec3 lerp(vec3 v0, vec3 v1, float t);
vec4 lerp(vec4 v0, vec4 v1, float t);

float log(float x);
float log10(float x);
float log2(float x);

// returns greater of x or y
float max(float x, float y);

// returns lesser of x or y
float min(float x, float y);

// multiplication of different types
//float mul(float x, float y); // 1
vec2 mul(float f, vec2 v); // 2
vec3 mul(float f, vec3 v);
vec4 mul(float f, vec4 v);
vec2 mul(vec2 v, float f); // 3
vec3 mul(vec3 v, float f);
vec4 mul(vec4 v, float f);
vec2 mul(vec2 v0, vec2 v1); // 4
vec3 mul(vec3 v0, vec3 v1);
vec4 mul(vec4 v0, vec4 v1);
float* mul(float f, mat2x2 m); // 5
float* mul(float f, mat3x3 m);
float* mul(float f, mat4x4 m);
float* mul(mat2x2 m, float f); // 6
float* mul(mat3x3 m, float f);
float* mul(mat4x4 m, float f);
float* mul(mat2x2 A, mat2x2 B); // 7
float* mul(mat3x3 A, mat3x3 B);
float* mul(mat4x4 A, mat4x4 B);
vec2 mul(vec2 v, mat2x2 m); // 8
vec3 mul(vec3 v, mat3x3 m);
vec4 mul(vec4 v, mat4x4 m);
vec2 mul(mat2x2 m, vec2 v); // 9
vec3 mul(mat3x3 m, vec3 v);
vec4 mul(mat4x4 m, vec4 v);

// return normalized vector (length of 1)
vec2 normalize(vec2 v);
vec3 normalize(vec3 v);
vec4 normalize(vec4 v);

// return x^y
float pow(float x, float y);

// convert degrees to radians
float radians(float d);

// reflect incident vector i across normal vector n
vec2 reflect(vec2 i, vec2 n);
vec3 reflect(vec3 i, vec3 n);
vec4 reflect(vec4 i, vec4 n);

// calculate refracted ray, using incoming ray r, normal vector n, and index of refraction
vec2 refract(vec2 r, vec2 n, float IR);
vec3 refract(vec3 r, vec3 n, float IR);
vec4 refract(vec4 r, vec4 n, float IR);

// returns 1/sqrt(x)
float rsqrt(float x);

// return sign of x (-1 or 1)
float sign(float x);

// sine of x
float sin(float x);

// returns hermite interpolation between 0 and 1, if x is between min and max. Otherwise clamps to 0/1
float smoothstep(float a, float b, float x);

// tangent of x
float tan(float x);

float *transpose(mat2x2 m);
float *transpose(mat3x3 m);
float *transpose(mat4x4 m);

#endif // _GFX_MATH_H