#include "GFX_Math.h"
#include "fastapprox.h"

// absolute value of each component of x
float g_abs(float x) {
  return x < 0 ? -x : x;
}
vec2 g_abs(vec2 v) {
  float x = g_abs(v.x);
  float y = g_abs(v.y);
  return vec2{x, y};
}
vec3 g_abs(vec3 v) {
  float x = g_abs(v.x);
  float y = g_abs(v.y);
  float z = g_abs(v.z);
  return vec3{x, y, z};
}
vec4 g_abs(vec4 v) {
  float x = g_abs(v.x);
  float y = g_abs(v.y);
  float z = g_abs(v.z);
  float w = g_abs(v.w);
  return vec4{x, y, z, w};
}

// returns true if all components are non-zero
bool all(float x) {
  return x != 0;
}
bool all(vec2 v) {
  if(v.x != 0 && v.y != 0) {
    return true;
  }
  else {
    return false;
  }
}
bool all(vec3 v) {
  if(v.x != 0 && v.y != 0 && v.z != 0) {
    return true;
  }
  else {
    return false;
  }
}
bool all(vec4 v) {
  if(v.x != 0 && v.y != 0 && v.z != 0 && v.w != 0) {
    return true;
  }
  else {
    return false;
  }
}
bool all(mat2x2 m) {
  if(m[0][0] != 0 && m[0][1] != 0 && m[1][0] != 0 && m[1][1] != 0) {
    return true;
  }
  else {
    return false;
  }
}
bool all(mat3x3 m) {
  if(m[0][0] != 0 && m[0][1] != 0 && m[0][2] != 0 && m[1][0] != 0 && m[1][1] != 0 && m[1][2] != 0
    && m[2][0] != 0 && m[2][1] != 0 && m[2][2] != 0) {
    return true;
  }
  else {
    return false;
  }
}
bool all(mat4x4 m) {
  if(m[0][0] != 0 && m[0][1] != 0 && m[0][2] != 0 && m[0][3] != 0 && m[1][0] != 0 && m[1][1] != 0
    && m[1][2] != 0 && m[1][3] != 0 && m[2][0] != 0 && m[2][1] != 0 && m[2][2] != 0 && m[2][3] != 0
    && m[3][0] != 0 && m[3][1] != 0 && m[3][2] != 0 && m[3][3] != 0) {
    return true;
  }
  else {
    return false;
  }
}

// arctangent of two values - uses built-in function
float atan2(float y, float x) {
  return atan2f(y, x);
}

// clamp value between min and max
float clamp(float x, float min, float max) {
  if(x > max) {
    return max;
  }
  else if(x < min) {
    return min;
  }
  else {
    return x;
  }
}

// clamp component-wise between 0 and 1
float clamp01(float x) {
  if(x > 1) {
    return 1;
  }
  else if(x < 0) {
    return 0;
  }
  else {
    return x;
  }
}
vec2 clamp01(vec2 v) {
  return vec2{
    clamp01(v.x),
    clamp01(v.y)
  };
}
vec3 clamp01(vec3 v) {
  return vec3{
    clamp01(v.x),
    clamp01(v.y),
    clamp01(v.z)
  };
}
vec4 clamp01(vec4 v) {
  return vec4{
    clamp01(v.x),
    clamp01(v.y),
    clamp01(v.z),
    clamp01(v.w)
  };
}

// cosine of each component
float cos(float x) {
  return sin(x + M_PI_2);
}

// cross product of two vectors
vec3 cross(vec3 a, vec3 b) {
  return vec3{
    a.y*b.z - a.z*b.y,
    a.z*b.x - a.x*b.z,
    a.x*b.y - a.y*b.x
  };
}

// convert from radians to degrees
float g_degrees(float r) {
  return r * M_RAD_TO_DEG;
}

// CREDIT: https://developer.download.nvidia.com/cg/determinant.html
// return determinant of square matrix
float determinant(mat2x2 m) {
  return m[0][0]*m[1][1] - m[0][1]*m[1][0];
}
float determinant(mat3x3 m) {
  return dot(
    vec3{m[0][0], m[0][1], m[0][2]},
      vec3{m[1][1], m[1][2], m[1][0]} * vec3{m[2][2], m[2][0], m[2][1]}
      - vec3{m[1][2], m[1][0], m[1][1]} * vec3{m[2][1], m[2][2], m[2][0]}
  );
}
float determinant(mat4x4 m) {
  return dot(
    vec4{1, -1, 1, -1} * vec4{m[0][0], m[0][1], m[0][2], m[0][3]},
      vec4{m[1][1], m[1][2], m[1][3], m[1][0]} *
        (vec4{m[2][2], m[2][3], m[2][0], m[2][1]} * vec4{m[3][3], m[3][0], m[3][1], m[3][2]}
        - vec4{m[2][3], m[2][0], m[2][1], m[2][2]} * vec4{m[3][2], m[3][3], m[3][0], m[3][1]})
      + vec4{m[1][2], m[1][3], m[1][0], m[1][1]} *
        (vec4{m[2][3], m[2][0], m[2][1], m[2][2]} * vec4{m[3][1], m[3][2], m[3][3], m[3][0]}
        - vec4{m[2][1], m[2][2], m[2][3], m[2][0]} * vec4{m[3][3], m[3][0], m[3][1], m[3][2]})
      + vec4{m[1][3], m[1][0], m[1][1], m[1][2]} *
        (vec4{m[2][1], m[2][2], m[2][3], m[2][0]} * vec4{m[3][2], m[3][3], m[3][0], m[3][1]}
        - vec4{m[2][2], m[2][3], m[2][0], m[2][1]} * vec4{m[3][1], m[3][2], m[3][3], m[3][0]})
  );
}

// CREDIT: https://developer.download.nvidia.com/cg/distance.html
// return distance between two points
float distance(vec2 v0, vec2 v1) {
  vec2 v = v1 - v0;
  return sqrtf(dot(v, v));
}
float distance(vec3 v0, vec3 v1) {
  vec3 v = v1 - v0;
  return sqrtf(dot(v, v));
}
float distance(vec4 v0, vec4 v1) {
  vec4 v = v1 - v0;
  return sqrtf(dot(v, v));
}

// CREDIT: https://developer.download.nvidia.com/cg/dot.html
// dot product of two vectors
float dot(vec2 a, vec2 b) {
  return a.x*b.x + a.y*b.y;
}
float dot(vec3 a, vec3 b) {
  return a.x*b.x + a.y*b.y + a.z*b.z;
}
float dot(vec4 a, vec4 b) {
  return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

// base-e exponential - uses built-in method
float exp(float x) {
  return fasterexp(x);
}

// base-2 exponential - uses built-in method
//float exp2(float x) {
//  return exp2f(x);
//}

// floating-point remainder of x/y - uses built-in method
float g_fmod(float a, float b) {
  float c = frac(g_abs(a/b))*g_abs(b);
  return (a < 0) ? -c : c;   /* if ( a < 0 ) c = 0-c */
}

// CREDIT: https://developer.download.nvidia.com/cg/frac.html
// returns the fractional part of x
float frac(float x) {
  return x - floorf(x);
}

// return length of vector
float length(vec2 v) {
  sqrtf((v.x*v.x)+(v.y*v.y));
}
float length(vec3 v) {
  sqrtf((v.x*v.x)+(v.y*v.y)+(v.z*v.z));
}
float length(vec4 v) {
  sqrtf((v.x*v.x)+(v.y*v.y)+(v.z*v.z)+(v.w*v.w));
}

// CREDIT: lerp implementation taken from Wikipedia
// return linear interpolation between v0 and v1 at point t
float lerp(float v0, float v1, float t) {
  return v0 + t * (v1 - v0);
}
vec2 lerp(vec2 v0, vec2 v1, float t) {
  float x = v0.x + t * (v1.x - v0.x);
  float y = v0.y + t * (v1.y - v0.y);
  return vec2{x, y};
}
vec3 lerp(vec3 v0, vec3 v1, float t) {
  float x = v0.x + t * (v1.x - v0.x);
  float y = v0.y + t * (v1.y - v0.y);
  float z = v0.z + t * (v1.z - v0.z);
  return vec3{x, y, z};
}
vec4 lerp(vec4 v0, vec4 v1, float t) {
  float x = v0.x + t * (v1.x - v0.x);
  float y = v0.y + t * (v1.y - v0.y);
  float z = v0.z + t * (v1.z - v0.z);
  float w = v0.w + t * (v1.w - v0.w);
  return vec4{x, y, z, w};
}

// log functions use built-in methods
float log(float x) {
  return fasterlog(x);
}
float log2(float x) {
  return fasterlog2(x);
}

// returns greater of x or y
float g_max(float x, float y) {
  return x >= y ? x : y;
}

// returns lesser of x or y
float g_min(float x, float y) {
  return x <= y ? x : y;
}

// multiplication of different types
// scale vector by float
vec2 mul(float f, vec2 v) {
  return f*v;
}
vec3 mul(float f, vec3 v) {
  return f*v;
}
vec4 mul(float f, vec4 v) {
  return f*v;
}
vec2 mul(vec2 v, float f) {
  return f*v;
}
vec3 mul(vec3 v, float f) {
  return f*v;
}
vec4 mul(vec4 v, float f) {
  return f*v;
}
// component-wise vector multiplication
vec2 mul(vec2 v0, vec2 v1) {
  return v0*v1;
}
vec3 mul(vec3 v0, vec3 v1) {
  return v0*v1;
}
vec4 mul(vec4 v0, vec4 v1) {
  return v0*v1;
}
// scale matrix by float
float* mul(float f, mat2x2 m) {
  static mat2x2 C = {
    {f*m[0][0], f*m[0][1]},
    {f*m[1][0], f*m[1][1]}
  };

  return *C;
}
float* mul(float f, mat3x3 m) {
  static mat3x3 C = {
    {f*m[0][0], f*m[0][1], f*m[0][2]},
    {f*m[1][0], f*m[1][1], f*m[1][2]},
    {f*m[2][0], f*m[2][1], f*m[2][2]}
  };

  return *C;
}
float* mul(float f, mat4x4 m) {
  static mat4x4 C = {
    {f*m[0][0], f*m[0][1], f*m[0][2], f*m[0][3]},
    {f*m[1][0], f*m[1][1], f*m[1][2], f*m[1][3]},
    {f*m[2][0], f*m[2][1], f*m[2][2], f*m[2][3]},
    {f*m[3][0], f*m[3][1], f*m[3][2], f*m[3][3]}
  };

  return *C;
}
float* mul(mat2x2 m, float f) {
  static mat2x2 C = {
    {f*m[0][0], f*m[0][1]},
    {f*m[1][0], f*m[1][1]}
  };

  return *C;
}
float* mul(mat3x3 m, float f) {
  static mat3x3 C = {
    {f*m[0][0], f*m[0][1], f*m[0][2]},
    {f*m[1][0], f*m[1][1], f*m[1][2]},
    {f*m[2][0], f*m[2][1], f*m[2][2]}
  };

  return *C;
}
float* mul(mat4x4 m, float f) {
  static mat4x4 C = {
    {f*m[0][0], f*m[0][1], f*m[0][2], f*m[0][3]},
    {f*m[1][0], f*m[1][1], f*m[1][2], f*m[1][3]},
    {f*m[2][0], f*m[2][1], f*m[2][2], f*m[2][3]},
    {f*m[3][0], f*m[3][1], f*m[3][2], f*m[3][3]}
  };

  return *C;
}
// matrix-matrix multiplication
float* mul(mat2x2 A, mat2x2 B) {
  static mat2x2 C = {
    {A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]},
    {A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]}
  };

  return *C;
}
float* mul(mat3x3 A, mat3x3 B) {
  static mat3x3 C = {
    {
      A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0], 
      A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1], 
      A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2]
    },
    {
      A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0], 
      A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1],
      A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2]
    },
    {
      A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0], 
      A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1],
      A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2]
    }
  };

  return *C;
}
float* mul(mat4x4 A, mat4x4 B) {
  static mat4x4 C = {
    {
      A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0] + A[0][3]*B[3][0], 
      A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1] + A[0][3]*B[3][1], 
      A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2] + A[0][3]*B[3][2],
      A[0][0]*B[0][3] + A[0][1]*B[1][3] + A[0][2]*B[2][3] + A[0][3]*B[3][3]
    },
    {
      A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0] + A[1][3]*B[3][0], 
      A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1] + A[1][3]*B[3][1],
      A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2] + A[1][3]*B[3][2],
      A[1][0]*B[0][3] + A[1][1]*B[1][3] + A[1][2]*B[2][3] + A[1][3]*B[3][3]
    },
    {
      A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0] + A[2][3]*B[3][0], 
      A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1] + A[2][3]*B[3][1],
      A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2] + A[2][3]*B[3][2],
      A[2][0]*B[0][3] + A[2][1]*B[1][3] + A[2][2]*B[2][3] + A[2][3]*B[3][3]
    },
    {
      A[3][0]*B[0][0] + A[3][1]*B[1][0] + A[3][2]*B[2][0] + A[3][3]*B[3][0], 
      A[3][0]*B[0][1] + A[3][1]*B[1][1] + A[3][2]*B[2][1] + A[3][3]*B[3][1],
      A[3][0]*B[0][2] + A[3][1]*B[1][2] + A[3][2]*B[2][2] + A[3][3]*B[3][2],
      A[3][0]*B[0][3] + A[3][1]*B[1][3] + A[3][2]*B[2][3] + A[3][3]*B[3][3]
    }
  };

  return *C;
}
// matrix-vector multiplication (linear mapping)
vec2 mul(vec2 v, mat2x2 m) {
  vec2 y;
  y.x = m[0][0]*v.x + m[0][1]*v.y;
  y.y = m[1][0]*v.x + m[1][1]*v.y;
  return y;
}
vec3 mul(vec3 v, mat3x3 m) {
  vec3 y;
  y.x = m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z;
  y.y = m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z;
  y.z = m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z;
  return y;
}
vec4 mul(vec4 v, mat4x4 m) {
  vec4 y;
  y.x = m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3]*v.w;
  y.y = m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3]*v.w;
  y.z = m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3]*v.w;
  y.w = m[3][0]*v.x + m[3][1]*v.y + m[3][2]*v.z + m[3][3]*v.w;
  return y;
}
vec2 mul(mat2x2 m, vec2 v) {
  vec2 y;
  y.x = m[0][0]*v.x + m[0][1]*v.y;
  y.y = m[1][0]*v.x + m[1][1]*v.y;
  return y;
}
vec3 mul(mat3x3 m, vec3 v) {
  vec3 y;
  y.x = m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z;
  y.y = m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z;
  y.z = m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z;
  return y;
}
vec4 mul(mat4x4 m, vec4 v) {
  vec4 y;
  y.x = m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3]*v.w;
  y.y = m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3]*v.w;
  y.z = m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3]*v.w;
  y.w = m[3][0]*v.x + m[3][1]*v.y + m[3][2]*v.z + m[3][3]*v.w;
  return y;
}

// CREDIT: https://developer.download.nvidia.com/cg/normalize.html
// return normalized vector (length of 1)
vec2 normalize(vec2 v) {
  return rsqrt(dot(v,v)) * v;
}
vec3 normalize(vec3 v) {
  return rsqrt(dot(v,v)) * v;
}
vec4 normalize(vec4 v) {
  return rsqrt(dot(v,v)) * v;
}

// return x^y - uses built-in method
float pow(float x, float y) {
  return fasterpow(x, y);
}

// convert degrees to radians
float g_radians(float d) {
  return d * M_DEG_TO_RAD;
}

// CREDTI: https://developer.download.nvidia.com/cg/reflect.html
// reflect incident vector i across normal vector n
vec2 reflect(vec2 i, vec2 n) {
  return i - 2.0 * n * dot(n,i);
}
vec3 reflect(vec3 i, vec3 n) {
  return i - 2.0 * n * dot(n,i);
}
vec4 reflect(vec4 i, vec4 n) {
  return i - 2.0 * n * dot(n,i);
}

// CREDIT: https://developer.download.nvidia.com/cg/refract.html
// calculate refracted ray, using incoming ray r, normal vector n, and index of refraction
vec2 refract(vec2 r, vec2 n, float IR) {
  float cosi = dot(-r, n);
  float cost2 = 1.0f - IR * IR * (1.0f - cosi*cosi);
  vec2 t = IR*r + ((IR*cosi - sqrtf(g_abs(cost2))) * n);
  return t * vec2{(cost2 > 0 ? 1.0f : 0), (cost2 > 0 ? 1.0f : 0)};
}
vec3 refract(vec3 r, vec3 n, float IR) {
  float cosi = dot(-r, n);
  float cost2 = 1.0f - IR * IR * (1.0f - cosi*cosi);
  vec3 t = IR*r + ((IR*cosi - sqrtf(g_abs(cost2))) * n);
  return t * vec3{(cost2 > 0 ? 1.0f : 0), (cost2 > 0 ? 1.0f : 0), (cost2 > 0 ? 1.0f : 0)};
}
vec4 refract(vec4 r, vec4 n, float IR) {
  float cosi = dot(-r, n);
  float cost2 = 1.0f - IR * IR * (1.0f - cosi*cosi);
  vec4 t = IR*r + ((IR*cosi - sqrtf(g_abs(cost2))) * n);
  return t * vec4{(cost2 > 0 ? 1.0f : 0), (cost2 > 0 ? 1.0f : 0), (cost2 > 0 ? 1.0f : 0), (cost2 > 0 ? 1.0f : 0)};
}

// returns 1/sqrt(x)
float rsqrt(float x) {
  return 1.0 / sqrtf(x);
}

// return sign of x (-1 or 1)
float sign(float x) {
  return x < 0 ? -1 : 1;
}

// CREDIT:
// https://stackoverflow.com/a/66868438 
// http://web.archive.org/web/20141220225551/http://forum.devmaster.net/t/fast-and-accurate-sine-cosine/9648
// (fast/approximate) sine of x
float sin(float x) {
  const float B = 4/M_PI;
  const float C = -4/(M_PI*M_PI);

  float y = B * x + C * x * g_abs(x);

  #ifdef EXTRA_PRECISION
  //  const float Q = 0.775;
      const float P = 0.225;

      y = P * (y * abs(y) - y) + y;   // Q * y + P * y * abs(y)
  #endif
  return y;
}

// CREDIT: https://developer.download.nvidia.com/cg/smoothstep.html
// returns hermite interpolation between 0 and 1, if x is between min and max. Otherwise clamps to 0/1
float smoothstep(float a, float b, float x) {
  float t = clamp01((x - a)/(b - a));
  return t*t*(3.0 - (2.0*t));
}

// CREDIT: http://www.ganssle.com/approx.htm
// uses the tan_56 function to compute (approximate) tangent up to 5.6 digits of accuracy
// tangent of x
float tan_56(float x) {
  const float c1 = -3.16783027;
  const float c2 = 0.134516124;
  const float c3 = -4.033321984;
  float x2 = x*x;
  return (x * (c1 + c2 * x2) / (c3 + x2));
}

// CREDIT: http://www.ganssle.com/approx.htm
// range reduction for tangent approximation, then call tangent
float tan(float x) {
  int octant;

  x = g_fmod(x, M_PI*2);
  octant = int(x/M_PI_4);
  switch(octant) {
    case 0: return        tan_56(x                     * M_4_OVER_PI);
    case 1: return  1.0 / tan_56((M_PI_2 - x)          * M_4_OVER_PI);
    case 2: return -1.0 / tan_56((x - M_PI_2)          * M_4_OVER_PI);
    case 3: return       -tan_56((M_PI - x)            * M_4_OVER_PI);
    case 4: return        tan_56((x - M_PI)            * M_4_OVER_PI);
    case 5: return  1.0 / tan_56((M_THREE_HALF_PI - x) * M_4_OVER_PI);
    case 6: return -1.0 / tan_56((x - M_THREE_HALF_PI) * M_4_OVER_PI);
    case 7: return       -tan_56((M_PI * 2)            * M_4_OVER_PI);
  }
}

float* transpose(mat2x2 m) {
  static mat2x2 C;

  C[0][0] = m[0][0];
  C[0][1] = m[1][0];
  C[1][0] = m[0][1];
  C[1][1] = m[1][1];
  
  return *C;
}
float* transpose(mat3x3 m) {
  static mat3x3 C;

  C[0][0] = m[0][0];
  C[0][1] = m[1][0];
  C[0][2] = m[2][0];
  C[1][0] = m[0][1];
  C[1][1] = m[1][1];
  C[1][2] = m[2][1];
  C[2][0] = m[0][2];
  C[2][1] = m[1][2];
  C[2][2] = m[2][2];

  return *C;
}
float* transpose(mat4x4 m) {
  static mat4x4 C;

  C[0][0] = m[0][0];
  C[0][1] = m[1][0];
  C[0][2] = m[2][0];
  C[0][3] = m[3][0];
  C[1][0] = m[0][1];
  C[1][1] = m[1][1];
  C[1][2] = m[2][1];
  C[1][3] = m[3][1];
  C[2][0] = m[0][2];
  C[2][1] = m[1][2];
  C[2][2] = m[2][2];
  C[2][3] = m[3][2];
  C[3][0] = m[0][3];
  C[3][1] = m[1][3];
  C[3][2] = m[2][3];
  C[3][3] = m[3][3];

  return *C;
}