#pragma once
#include <vector>

template <typename T>
struct vec2 {
	T x, y;
	vec2() : x(0), y(0) {}
	vec2(T x, T y) : x(x), y(y) {}

	vec2 operator+(vec2 const& other) const {
		return vec2(x + other.x, y + other.y);
	}

	vec2 operator-(vec2 const& other) const {
		return vec2(x - other.x, y - other.y);
	}

	vec2 operator*(float k) const {
		return vec2<T>(x * k, y * k);
	}

	vec2 operator/(float k) const {
		return vec2<T>(x / k, y / k);
	}

	vec2& operator+=(vec2 const& other) {
		x += other.x;
		y += other.y;
		return *this;
	}
};

template <typename T>
struct vec3 {
	T x, y, z;
	vec3() : x(0), y(0), z(0) {}
	vec3(T x, T y, T z) : x(x), y(y), z(z) {}

	vec3 operator+(vec3 const& other) const {
		return vec3(x + other.x, y + other.y, z + other.z);
	}

	vec3 operator-(vec3 const& other) const {
		return vec3(x - other.x, y - other.y, z - other.z);
	}

	vec3 operator-() const {
		return vec3(-x, -y, -z);
	}

	vec3 operator*(float k) const {
		return vec3<T>(x * k, y * k, z * k);
	}

	vec3 operator/(float k) const {
		return vec3<T>(x / k, y / k, z / k);
	}

	vec3& operator+=(vec3 const& other) {
		x += other.x;
		y += other.y;
		z += other.z;
		return *this;
	}

	float abs() const {
		return sqrt(x * x + y * y + z * z);
	}

	vec3 normalize() {
		float len = abs();
		x /= len;
		y /= len;
		z /= len;
		return *this;
	}

	/*
	vec3& operator=(vec3 const& other) {
	x = other.x;
	y = other.y;
	z = other.z;
	return *this;
	}*/
};

template <typename T>
struct vec4 {
	T x, y, z, w;
	vec4() : x(0), y(0), z(0), w(0) {}
	vec4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
	vec4(vec3<T> v, float w) : x(v.x), y(v.y), z(v.z), w(w) {}

	template <typename T>
	operator vec3<T>() {
		return vec3<T>(x, y, z);
	}
};

typedef vec2<int> ivec2;
typedef vec3<int> ivec3;
typedef vec2<float> fvec2;
typedef vec3<float> fvec3;
typedef vec4<float> fvec4;

fvec3 operator*(std::vector<std::vector<float>> matrix, fvec3 v) {
	return {
		matrix[0][0] * v.x + matrix[0][1] * v.y + matrix[0][2] * v.z,
		matrix[1][0] * v.x + matrix[1][1] * v.y + matrix[1][2] * v.z,
		matrix[2][0] * v.x + matrix[2][1] * v.y + matrix[2][2] * v.z
	};
}

fvec4 operator*(std::vector<std::vector<float>> matrix, fvec4 v) {
	return {
		matrix[0][0] * v.x + matrix[0][1] * v.y + matrix[0][2] * v.z + matrix[0][3] * v.w,
		matrix[1][0] * v.x + matrix[1][1] * v.y + matrix[1][2] * v.z + matrix[1][3] * v.w,
		matrix[2][0] * v.x + matrix[2][1] * v.y + matrix[2][2] * v.z + matrix[2][3] * v.w,
		matrix[3][0] * v.x + matrix[3][1] * v.y + matrix[3][2] * v.z + matrix[3][3] * v.w
	};
}

float dotProduct(fvec3 a, fvec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

fvec3 crossProduct(fvec3 a, fvec3 b) {
	return fvec3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

std::vector<std::vector<float>> operator*(std::vector<std::vector<float>> const& a, std::vector<std::vector<float>> const& b) {
	std::vector<std::vector<float>> res(a.size(), std::vector<float>(b[0].size()));
	for (size_t i = 0; i < a.size(); i++) {
		for (size_t j = 0; j < b[0].size(); j++) {
			for (size_t k = 0; k < a[0].size(); k++) {
				res[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	return res;
}

fvec3 interpolatedVec(fvec3 const& a, fvec3 const& b, float interpolationCoef) {
	return a + (b - a) * interpolationCoef;
}

fvec2 interpolatedVec2(fvec2 const& a, fvec2 const& b, float interpolationCoef) {
	return a + (b - a) * interpolationCoef;
}