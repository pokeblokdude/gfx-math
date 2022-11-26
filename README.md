# GFX Math
This is a simple math library that implements common graphics functions. I made it for use with Arduino microcontrollers, but I'm sure it could be mildly useful in general programming as well. The initial list of functions was taken from HLSL documentation.

The main reason I'm documenting this so well is for me to look at lol (also it was just fun to make).

## Installation
To use this in the Arduino IDE, simply place the files into a new folder in the `libraries` directory. On Windows, this is `C:\Users\Your_Name\Documents\Arduino\libaries`. I'm sure it's similar on Mac and Linux but I have no way of checking.

## Types/Structs

### Constants (that aren't from <math.h>)
**M_4_OVER_PI** - 4 / pi  
**M_THREE_HALF_PI** - 3pi / 2  
**M_RAD_TO_DEG** - 180 / pi  
**M_DEG_TO_RAD** - pi / 180

### Vectors
`+`, `-`, and `*` operators are defined for all vector types.  
`*` is also defined for **vector*float**, **float*vector**, and the integer counterparts.

**vec2**  
Vector with 2 floats, `x`, `y`

**vec2i**  
Vector with 2 ints, `x`, `y`

**vec3**  
Vector with 3 floats, `x`, `y`, `z`

**vec3i**  
Vector with 3 ints, `x`, `y`, `z`

**vec4**  
Vector with 4 floats, `x`, `y`, `z`, `w`

**vec4i**  
Vector with 4 ints, `x`, `y`, `z`, `w`

### Matrices
**mat2x2**  
2x2 matrix with 4 floats  
`m00`, `m01`,  
`m10`, `m11`

**mat3x3**  
3x3 matrix with 9 floats  
`m00`, `m01`, `m02`,  
`m10`, `m11`, `m12`,  
`m20`, `m21`, `m22`

**mat4x4**  
4x4 matrix with 16 floats  
`m00`, `m01`, `m02`, `m03`,  
`m10`, `m11`, `m12`, `m13`,  
`m20`, `m21`, `m22`, `m23`,  
`m30`, `m31`, `m32`, `m33`


## Functions
Some functions have a `g_` prefix because they were conflicting with the built-in math library.

**g_abs(x)**  
returns the component-wise absolute value of x   
`x`: float, vec2, vec3, vec4  
`return`:  same type as `x`

**all(x)**  
returns true if all components of x are non-zero  
`x`: float, vec2, vec3, vec4, mat2x2, mat3x3, mat4x4  
`return`:  bool

**g_atan2(y, x)**  
to be honest I still don't really understand what this does  
`y`: float  
`x`: float  
`return`:  float

**clamp(x, min, max)**
clamps x between min and max  
`x`: float   
`min`: float  
`max`: float   
`return`: float

**clamp01(x)**  
component-wise clamps x between 0 and 1  
`x`: float, vec2, vec3, vec4  
`return`:  same type as `x`

**g_cos(x)**  
returns the cosine of x, using a fast-approximate sine function  
`x`: float  
`return`: float

**cross(a, b)**  
returns the cross-product of vectors a and b  
`a`: vec3  
`b`: vec3   
`return`: vec3

**g_degrees(r)**  
converts r from radians to degrees  
`x`: float  
`return`: float

**determinant(m)**  
returns the determinant of a square matrix  
`x`: mat2x2, mat3x3, mat4x4  
`return`: float

**distance(v0, v1)**  
returns the euclidean distance between vectors v0 and v1  
`v0`: vec2, vec3, vec4  
`v1`: same type as `v0`  
`return`: float

**dot(a, b)**  
returns the dot-product of vectors a and b  
`a`: vec2, vec3, vec4  
`b`: same type as `a`  
`return`:  same type as `x`

**exp(x)**  
returns e^x  
`x`: float  
`return`: float

**g_fmod(a, b)**  
returns the floating-point remainder of a/b  
`a`: float  
`b`: float  
`return`: float

**frac(x)**  
returns the fractional part of x  
`x`: float  
`return`: float

**inverse(M)**  
returns the inverse of a matrix M  
`M`: mat2x2, mat3x3, mat4x4  
`return`: same type as `M`

**invLerp(a, b, value)**  
returns a value t, given two points a and b, and a value (inverse of lerp)  
`a`: float  
`b`: float  
`value`: float  
`return`: float

**length(v)**  
returns the length (magnitude) of a vector  
`v`: vec2, vec3, vec4  
`return`: float

**lerp(v0, v1, t)**  
returns a linear interpolation between v0 and v1, according to t  
`v0`: float, vec2, vec3, vec4  
`v1`: same type as `v0`  
`t`: float  
`return`:  same type as `v0`

**g_log(x)**  
returns the natural log of x  
`x`: float  
`return`: float

**g_log2(x)**  
returns log (base-2) of x  
`x`: float  
`return`: float

**g_max(x, y)**  
returns the greater of x and y  
`x`: float  
`y`: float  
`return`: float

**g_min(x, y)**  
returns the lesser of x and y  
`x`: float  
`y`: float  
`return`: float

Here comes a doosey...  

**mul(f, v)**, **mul(v, f)**  
returns vector v scaled by f  
`v`: vec2, vec3, vec4  
`f`: float  
`return`:  same type as `v`

**mul(v0, v1)**  
returns the component-wise multiple of vectors v0 and v1  
`v0`: vec2, vec3, vec4  
`v1`: same type as `v0`  
`return`:  same type as `v0`

**mul(f, M)**, **mul(M, f)**  
returns a matrix M scaled by f  
`M`: mat2x2, mat3x3, mat4x4  
`f`: float  
`return`:  same type as `M`

**mul(A, B)**  
returns the matrix multiple of A*B  
`A`: mat2x2, mat3x3, mat4x4  
`B`: same type as `A`  
`return`:  same type as `A`

**mul(M, v)**, **mul(v, M)**  
returns the matrix-vector multiple of M*v  
`M`: mat2x2, mat3x3, mat4x4  
`v`: vector, where vecN matches matNxN (ie, vec3 and mat3x3)  
`return`:  same type as `M`

Now that's out of the way...

**normalize(v)**  
returns the normalized vector v (length of 1)  
`v`: vec2, vec3, vec4  
`return`:  same type as `v`

**g_pow(x, y)**  
returns x^y  
`x`: float  
`y`: float  
`return`: float

**g_radians(d)**  
converts d from degrees to radians  
`d`: float  
`return`: float

**reflect(i, n)**  
returns an incident vector i, reflected across a normal vector n  
`i`: vec2, vec3, vec4  
`n`: same type as `i`  
`return`:  same type as `i`

**refract(r, n, IR)**  
returns a ray r, refracted according to a normal vector n and index of refraction (to be honest I don't know if this one works)  
`r`: vec2, vec3, vec4  
`n`: same type as `r`  
`IR`: float  
`return`:  same type as `r`

**rsqrt(x)**  
returns 1/sqrt(x)  
`x`: float  
`return`: float

**sign(x)**  
returns the sign of x (-1 or 1)  
`x`: float  
`return`: float

**g_sin(x)**  
returns the (approximated) sine of x  
`x`: float  
`return`: float

**smoothstep(a, b, x)**  
returns a hermite interpolation between 0 and 1, clamped to (a, b)  
`a`: float  
`b`: float  
`x`: float  
`return`:  float

**g_tan(x)**  
returns the (approximate) tangent of x  
`x`: float  
`return`: float

**transpose(m)**  
returns the transpose of a matrix  
`m`: mat2x2, mat3x3, mat4x4  
`return`:  same type as `m`