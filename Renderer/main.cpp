#define _USE_MATH_DEFINES
#include <windows.h>
#include <gdiplus.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <ctime>
#include <iostream>
#include <cmath>
#include <chrono>
#include <type_traits>
#include <memory>

#include "vec.h"

using namespace Gdiplus;
#pragma comment (lib,"Gdiplus.lib")

typedef unsigned char BYTE;

int const WIDTH = 800;
int const HEIGHT = 600;
float const VIEWPORT_WIDTH = 1;
float const VIEWPORT_HEIGHT = VIEWPORT_WIDTH / WIDTH * HEIGHT;
float const VIEWPORT_Z = 1;

char data[WIDTH * HEIGHT * 3];
float depthBuf[WIDTH][HEIGHT];
Bitmap* myImage;

struct texture {
public:
	int width;
	int height;
	std::shared_ptr<uint8_t> data;

	texture(Bitmap& bitmap) {
		BitmapData bitmapData;
		width = bitmap.GetWidth();
		height = bitmap.GetHeight();
		bitmap.LockBits(&Rect(0, 0, width, height), ImageLockModeRead, PixelFormat24bppRGB, &bitmapData);
		uint8_t* buf = new uint8_t[width * height * 3];
		memcpy(buf, bitmapData.Scan0, width * height * 3);
		data = std::shared_ptr<uint8_t>(buf, std::default_delete<uint8_t[]>());
	}

	texture() : width(0), height(0), data(nullptr) {}
};

struct triangle {
public:
	int vs[3];
	ivec3 color;
	fvec3 normals[3];
	texture diffuse_map;
	fvec2 uvs[3];

	triangle() = default;

	triangle(int a, int b, int c,
			 ivec3 const& color,
			 fvec3 const& na, fvec3 const& nb, fvec3 const& nc,
			 texture diffuse_map,
			 fvec2 const& uva, fvec2 const& uvb, fvec2 const& uvc)
	: color(color), diffuse_map(diffuse_map) {
		vs[0] = a;
		vs[1] = b;
		vs[2] = c;

		normals[0] = na;
		normals[1] = nb;
		normals[2] = nc;

		uvs[0] = uva;
		uvs[1] = uvb;
		uvs[2] = uvc;
	}
};

void setPixel(int x, int y, byte r, byte g, byte b) {
#ifdef _DEBUG
	if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) {
		fprintf(stdout, "Out of drawing window: setPixel(%d, %d, %d, %d, %d)\n", x, y, r, g, b);
	} else {
#endif
		int const pos = y * WIDTH * 3 + x * 3;
		data[pos] = b;
		data[pos + 1] = g;
		data[pos + 2] = r;
#ifdef _DEBUG
	}
#endif
}

ivec3 getPixel(texture const& diffuse_map, fvec2 const& pos) {
	uint8_t* buf = diffuse_map.data.get();
	int h = diffuse_map.height;
	int w = diffuse_map.width;
	
	int offset = ((h - (int)(pos.y * (h - 1)) - 1) * w + (int)(pos.x * (w - 1))) * 3;
	return ivec3(buf[offset + 2], buf[offset + 1], buf[offset]);
}

struct model {
	std::vector<fvec3> vertexes;
	std::vector<triangle> triangles;

	model() = default;

	model(std::vector<fvec3> vertexes, std::vector<triangle> triangles)
		: vertexes(std::move(vertexes))
		, triangles(std::move(triangles))
	{}
};

struct modelInstance {
private:
	std::vector<std::vector<float>> getRotationMatrix(fvec3 rotation) {
		std::vector<std::vector<float>> rx{
			{ 1,	0,				0,					0 },
			{ 0, cos(rotation.x),	-sin(rotation.x),	0 },
			{ 0, sin(rotation.x),	cos(rotation.x),	0 },
			{ 0,	0,				0,					1 }
		};
		std::vector<std::vector<float>> ry{
			{ cos(rotation.y),	0,	sin(rotation.y),	0 },
			{ 0,				1,	0,					0 },
			{ -sin(rotation.y),	0,	cos(rotation.y),	0 },
			{ 0,				0,	0,					1 }
		};
		std::vector<std::vector<float>> rz{
			{ cos(rotation.z),	-sin(rotation.z),	0,	0 },
			{ sin(rotation.z),	cos(rotation.z),	0,	0 },
			{ 0,				0,					1,	0 },
			{ 0,				0,					0,	1 }
		};
		return rx * ry * rz;
	}

	std::vector<std::vector<float>> getScaleMatrix(float scale) {
		std::vector<std::vector<float>> res{
			{ scale,	0,		0,		0 },
			{ 0,		scale,	0,		0 },
			{ 0,		0,		scale,	0 },
			{ 0,		0,		0,		1 }
		};
		return res;
	}

public:
	std::shared_ptr<model> m;
	std::vector<std::vector<float>> scale;
	std::vector<std::vector<float>> rotation;
	std::vector<std::vector<float>> pos;
	std::vector<std::vector<float>> transform;

	modelInstance(std::shared_ptr<model> m, float scale, fvec3 rotation, fvec3 pos) : m(m) {
		this->rotation = getRotationMatrix(rotation);
		this->scale = getScaleMatrix(scale);

		this->pos = {
			{ 1, 0, 0, pos.x },
			{ 0, 1, 0, pos.y },
			{ 0, 0, 1, pos.z },
			{ 0, 0, 0, 1 }
		};

		this->transform = this->pos * this->scale * this->rotation;
	}

	void rotate(fvec3 angles) {
		std::vector<std::vector<float>> rotationMatrix = getRotationMatrix(angles);
		this->rotation = this->rotation * rotationMatrix;
		this->transform = this->transform * rotationMatrix;
	}

	void rescale(float sz) {
		this->scale = getScaleMatrix(sz);
		this->transform = this->pos * this->scale * this->rotation;
	}
};

struct directionLight {
	fvec3 direction;
	float intensity;

	directionLight(fvec3 direction, float intensity) : direction(direction), intensity(intensity) {}
};

struct pointLight {
	fvec3 pos;
	float intensity;

	pointLight(fvec3 pos, float intensity) : pos(pos), intensity(intensity) {}
};

struct lightsContainer {
	float ambientLightIntensity = 0.2;
	std::vector<directionLight> directionLights;
	std::vector<pointLight> pointLights;
};

struct scene {
	std::vector<modelInstance> modelInstances;
	lightsContainer lights;
};

struct camera {
	std::vector<std::vector<float>> pos;
	std::vector<std::vector<float>> rotation;
	std::vector<std::vector<float>> transform;

	camera(fvec3 pos, fvec3 rotation) {
		this->pos = {
			{ 1, 0, 0, -pos.x },
			{ 0, 1, 0, -pos.y },
			{ 0, 0, 1, -pos.z },
			{ 0, 0, 0, 1 }
		};

		std::vector<std::vector<float>> rx{
			{ 1,	0,					0,					0 },
			{ 0, cos(rotation.x),	sin(rotation.x),	0 },
			{ 0, -sin(rotation.x),	cos(rotation.x),	0 },
			{ 0,	0,					0,					1 }
		};
		std::vector<std::vector<float>> ry{
			{ cos(rotation.y),	0,	-sin(rotation.y),	0 },
			{ 0,					1,	0,					0 },
			{ sin(rotation.y),	0,	cos(rotation.y),	0 },
			{ 0,					0,	0,					1 }
		};
		std::vector<std::vector<float>> rz{
			{ cos(rotation.z),	sin(rotation.z),	0,	0 },
			{ -sin(rotation.z),  cos(rotation.z),	0,	0 },
			{ 0,					0,					1,	0 },
			{ 0,					0,					0,	1 }
		};
		this->rotation = rx * ry * rz;
		this->transform = this->rotation * this->pos;
	}
};

ivec3 limitColor(ivec3 const& color) {
	return ivec3(min(color.x, 255), min(color.y, 255), min(color.z, 255));
}

float computeIlluminationByDirection(fvec3 pl, fvec3 normal, fvec3 v, float lightIntensity, float s) {
	float nDotPl = dotProduct(pl, normal);
	float res = 0;
	float debug1, debug2;
	if (nDotPl > 0) {
		res = nDotPl / (pl.abs() * normal.abs()) * lightIntensity; // diffuse
		debug1 = res;
	}
	fvec3 reflected = normal * (dotProduct(normal, pl) / normal.abs() * 2) - pl;
	float cosBeta = dotProduct(v, reflected) / (v.abs() * reflected.abs());
	if (cosBeta > 0) {
		res += pow(cosBeta, s) * lightIntensity; // specular
		debug2 = pow(cosBeta, s) * lightIntensity;
	}
	return res;
}

// p - point in the scene
// v - vector from p to camera
// s - specular coefficient
float computeIllumination(fvec3 p, fvec3 normal, lightsContainer const& lights, camera const& cam, float s) {
	// pl - vector from p to light.center
	float res = lights.ambientLightIntensity;
	for (pointLight const& light : lights.pointLights) {
		res += computeIlluminationByDirection(light.pos - p, normal, -p, light.intensity, s);
	}
	for (directionLight const& light : lights.directionLights) {
		res += computeIlluminationByDirection(light.direction, normal, -p, light.intensity, s);
	}
	//return 1;
	return res;
}

template <typename T>
std::vector<T> interpolate(int x0, T y0, int x1, T y1) {
	std::vector<T> res;
	size_t sz = x1 - x0 + 1;
	res.reserve(sz);
	T a = (y1 - y0) / static_cast<float>(x1 - x0);
	T y = y0;
	for (int x = x0; x <= x1; x++) {
		res.push_back(y);
		y += a;
	}
	return res;
}

ivec2 viewPortToCanvas(float x, float y) {
	// x : [-VIEWPORT_WIDTH / 2 ... VIEWPORT_WIDTH / 2] --> [0 .. WIDTH - 1]
	// y : [-VIEWPORT_HEIGHT / 2 ... VIEWPORT_HEIGHT / 2] --> [0 .. HEIGHT - 1]
	// adding 0.5f for correct rounding
	return ivec2((WIDTH - 1) / VIEWPORT_WIDTH * x + WIDTH / 2,
				 (HEIGHT - 1) / VIEWPORT_HEIGHT * y + HEIGHT / 2);
}

fvec2 canvasToViewPort(int x, int y) {
	return fvec2(VIEWPORT_WIDTH / (WIDTH - 1) * x - VIEWPORT_WIDTH / 2,
				 VIEWPORT_HEIGHT / (HEIGHT - 1) * y - VIEWPORT_HEIGHT / 2);
}

fvec3 unprojectVertex(fvec2 p, float invZ) {
	return fvec3(p.x / (VIEWPORT_Z * invZ), p.y / (VIEWPORT_Z * invZ), 1 / invZ);
}

ivec2 projectVertex(fvec4 v) {
	// WIDTH * VIEWPORT_Z * v.x / (v.z * VIEWPORT_WIDTH) + WIDTH / 2.f
	return viewPortToCanvas(v.x * VIEWPORT_Z / v.z, v.y * VIEWPORT_Z / v.z);
}

void drawLine(ivec2 p0, ivec2 p1, ivec3 color) {
	if (abs(p1.x - p0.x) > abs(p1.y - p0.y)) {
		if (p0.x > p1.x) {
			std::swap(p0, p1);
		}
		std::vector<float> ys = interpolate(p0.x, (float)p0.y, p1.x, (float)p1.y);
		for (int x = p0.x; x <= p1.x; x++) {
			setPixel(x, round(ys[x - p0.x]), color.x, color.y, color.z); // todo round(ys[...])?
		}
	} else {
		if (p0.y > p1.y) {
			std::swap(p0, p1);
		}
		std::vector<float> xs = interpolate(p0.y, (float)p0.x, p1.y, (float)p1.x);
		for (int y = p0.y; y <= p1.y; y++) {
			setPixel(xs[y - p0.y], y, color.x, color.y, color.z); // todo round(xs[...])?
		}
	}
}

void drawWireframeTriangle(ivec2 p0, ivec2 p1, ivec2 p2, ivec3 color) {
	drawLine(p0, p1, color);
	drawLine(p1, p2, color);
	drawLine(p2, p0, color);
}

void drawFilledTriangle(ivec2 p0, ivec2 p1, ivec2 p2, ivec3 color) {
	if (p1.y < p0.y) {
		std::swap(p1, p0);
	}
	if (p2.y < p0.y) {
		std::swap(p0, p2);
	}
	if (p2.y < p1.y) {
		std::swap(p1, p2);
	}
	// x01 actually
	std::vector<float> x012 = interpolate(p0.y, (float)p0.x, p1.y, (float)p1.x); // todo without new ivec2? x <--> y
	std::vector<float> x12 = interpolate(p1.y, (float)p1.x, p2.y, (float)p2.x);
	std::vector<float> x02 = interpolate(p0.y, (float)p0.x, p2.y, (float)p2.x);
	x012.pop_back();
	x012.insert(x012.end(), x12.begin(), x12.end()); // now truly x012
	const size_t m = x02.size() / 2;
	if (x02[m] > x012[m]) {
		x02.swap(x012);
	}
	// horizontal line from x02[i] to x012[i]
	for (int y = p0.y; y <= p2.y; y++) {
		for (int x = x02[y - p0.y]; x <= x012[y - p0.y]; x++) {
			setPixel(x, y, color.x, color.y, color.z);
		}
	}
}

void drawShadedTriangle(ivec2 p0, float h0, ivec2 p1, float h1, ivec2 p2, float h2, ivec3 color) {
	if (p1.y < p0.y) {
		std::swap(p1, p0);
		std::swap(h1, h0);
	}
	if (p2.y < p0.y) {
		std::swap(p0, p2);
		std::swap(h0, h2);
	}
	if (p2.y < p1.y) {
		std::swap(p1, p2);
		std::swap(h1, h2);
	}
	// x01 actually
	std::vector<float> x012 = interpolate(p0.y, (float)p0.x, p1.y, (float)p1.x); // todo without new ivec2? x <--> y
	std::vector<float> h012 = interpolate(p0.y, h0, p1.y, h1);
	std::vector<float> x12 = interpolate(p1.y, (float)p1.x, p2.y, (float)p2.x);
	std::vector<float> h12 = interpolate(p1.y, h1, p2.y, h2);
	std::vector<float> x02 = interpolate(p0.y, (float)p0.x, p2.y, (float)p2.x);
	std::vector<float> h02 = interpolate(p0.y, h0, p2.y, h2);
	x012.pop_back();
	h012.pop_back();
	x012.insert(x012.end(), x12.begin(), x12.end()); // now truly x012
	h012.insert(h012.end(), h12.begin(), h12.end());
	const int m = x02.size() / 2;
	if (x02[m] > x012[m]) {
		x02.swap(x012);
		h02.swap(h012);
	}
	// horizontal line from x02[i] to x012[i]
	for (int y = p0.y; y <= p2.y; y++) {
		int xl = x02[y - p0.y];
		int xr = x012[y - p0.y];
		std::vector<float> h = interpolate(xl, h02[y - p0.y], xr, h012[y - p0.y]);
		for (int x = xl; x <= xr; x++) {
			setPixel(x, y, color.x * h[x - xl], color.y * h[x - xl], color.z * h[x - xl]);
		}
	}
}

// z_i actually is 1/z_i
// uv_i actually is uv_i/z_i
void drawTriangleWidthDepthBuffer(ivec2 p0, float z0, fvec3 n0, fvec2 uv0,
								  ivec2 p1, float z1, fvec3 n1, fvec2 uv1,
								  ivec2 p2, float z2, fvec3 n2, fvec2 uv2,
								  texture diffuse_map, ivec3 color,
								  lightsContainer const& lights, camera const& cam) {
	if (p1.y < p0.y) {
		std::swap(p1, p0);
		std::swap(z1, z0);
		std::swap(n1, n0);
		std::swap(uv1, uv0);
	}
	if (p2.y < p0.y) {
		std::swap(p0, p2);
		std::swap(z0, z2);
		std::swap(n0, n2);
		std::swap(uv0, uv2);
	}
	if (p2.y < p1.y) {
		std::swap(p1, p2);
		std::swap(z1, z2);
		std::swap(n1, n2);
		std::swap(uv1, uv2);
	}
	// x01 actually
	std::vector<float> x012 = interpolate(p0.y, (float)p0.x, p1.y, (float)p1.x); // todo without new ivec2? x <--> y
	std::vector<float> z012 = interpolate(p0.y, z0, p1.y, z1);
	std::vector<fvec3> n012 = interpolate(p0.y, n0, p1.y, n1);
	std::vector<fvec2> uv012 = interpolate(p0.y, uv0, p1.y, uv1);

	std::vector<float> x12 = interpolate(p1.y, (float)p1.x, p2.y, (float)p2.x);
	std::vector<float> z12 = interpolate(p1.y, z1, p2.y, z2);
	std::vector<fvec3> n12 = interpolate(p1.y, n1, p2.y, n2);
	std::vector<fvec2> uv12 = interpolate(p1.y, uv1, p2.y, uv2);

	std::vector<float> x02 = interpolate(p0.y, (float)p0.x, p2.y, (float)p2.x);
	std::vector<float> z02 = interpolate(p0.y, z0, p2.y, z2);
	std::vector<fvec3> n02 = interpolate(p0.y, n0, p2.y, n2);
	std::vector<fvec2> uv02 = interpolate(p0.y, uv0, p2.y, uv2);

	x012.pop_back();
	z012.pop_back();
	n012.pop_back();
	uv012.pop_back();

	x012.insert(x012.end(), x12.begin(), x12.end()); // now truly x012
	z012.insert(z012.end(), z12.begin(), z12.end());
	n012.insert(n012.end(), n12.begin(), n12.end());
	uv012.insert(uv012.end(), uv12.begin(), uv12.end());

	const int m = x02.size() / 2;
	if (x02[m] > x012[m] || x02.size() == 2 && x02[1] == x012[1] && x02[0] > x012[0]) {
		x02.swap(x012);
		z02.swap(z012);
		n02.swap(n012);
		uv02.swap(uv012);
	}
	// horizontal line from x02[i] to x012[i]
	
	for (int y = p0.y; y < p2.y; y++) { // <= ? (we never draw the highest row
		int xl = x02[y - p0.y];
		int xr = x012[y - p0.y];
		std::vector<float> z = interpolate(xl, z02[y - p0.y], xr, z012[y - p0.y]);
		std::vector<fvec3> n = interpolate(xl, n02[y - p0.y], xr, n012[y - p0.y]);
		std::vector<fvec2> uv = interpolate(xl, uv02[y - p0.y], xr, uv012[y - p0.y]);
		// todo we never draw the rightest column
		// todo ... and one before the rightest (why?)
		/*if (xr == WIDTH - 1) {
			xr++;
		}*/
		for (int x = xl; x < xr; x++) {
			if (z[x - xl] > depthBuf[x][y]) {
				/*if (depthBuf[x][y] != 0) {
					//printf("%d %d\n", x, y);
					setPixel(x, y, 0, 0, 0);
					k++;
					printf("%d %d %d\n", k, x, y);
					continue;
				}*/
				depthBuf[x][y] = z[x - xl];
				
				if (diffuse_map.data != nullptr) {
					color = getPixel(diffuse_map, uv[x - xl] / z[x - xl]);
				}
				//if (x == xl)
				//std::cout << n[x - xl].x << ' ' << n[x - xl].y << ' ' << n[x - xl].z << std::endl;
				ivec3 curColor = limitColor(color * computeIllumination(
					unprojectVertex(canvasToViewPort(x, y), depthBuf[x][y]),
					n[x - xl], lights, cam, 50));

				setPixel(x, y, curColor.x, curColor.y, curColor.z);
			}
		}
	}
}

ivec3 const red = ivec3(255, 0, 0);
ivec3 const green = ivec3(0, 255, 0);
ivec3 const blue = ivec3(0, 0, 255);
ivec3 const yellow = ivec3(255, 255, 0);
ivec3 const purple = ivec3(255, 0, 255);
ivec3 const cyan = ivec3(0, 255, 255);

void renderWireframeTriangle(triangle const& t, std::vector<ivec2> const& projected) {
	drawWireframeTriangle(
		projected[t.vs[0]],
		projected[t.vs[1]],
		projected[t.vs[2]],
		t.color
	);
}

void renderTriangle(triangle const& t, model const& m, std::vector<ivec2> const& projected,
					lightsContainer const& lights, camera const& cam) {
	drawTriangleWidthDepthBuffer(
		projected[t.vs[0]], 1 / m.vertexes[t.vs[0]].z, t.normals[0], t.uvs[0] / m.vertexes[t.vs[0]].z,
		projected[t.vs[1]], 1 / m.vertexes[t.vs[1]].z, t.normals[1], t.uvs[1] / m.vertexes[t.vs[1]].z,
		projected[t.vs[2]], 1 / m.vertexes[t.vs[2]].z, t.normals[2], t.uvs[2] / m.vertexes[t.vs[2]].z,
		t.diffuse_map, t.color, lights, cam
	);
	//renderWireframeTriangle(t, projected);
	/*drawWireframeTriangle(
		projected[t.vs[0]],
		projected[t.vs[1]],
		projected[t.vs[2]],
		ivec3(0, 0, 0)
	);*/
}

struct clippingPlane {
	fvec3 n;
	float d;

	clippingPlane(fvec3 n, float d) : n(n), d(d) {}

	inline float dst(fvec3 p) const {
		return n.x * p.x + n.y * p.y + n.z * p.z + d;
	}

	float intersectionCoef(fvec3 const& a, fvec3 const& b) const {
		return (-d - dotProduct(n, a)) / dotProduct(n, b - a);
	}
	
	fvec3 intersection(fvec3 const& a, fvec3 const& b, float intCoef) const {
		return a + (b - a) * intCoef;
	}

	fvec3 intersection(fvec3 const& a, fvec3 const& b) const {
		return a + (b - a) * intersectionCoef(a, b);
	}
};

std::vector<clippingPlane> cPlanes = {
	{fvec3(0, 0, 1), 1}, // near
	{fvec3(-2 * VIEWPORT_Z / VIEWPORT_WIDTH, 0, 1).normalize(), 0}, // right
	{fvec3(2 * VIEWPORT_Z / VIEWPORT_WIDTH, 0, 1).normalize(), 0}, // left
	{fvec3(0, -1, VIEWPORT_HEIGHT / (VIEWPORT_Z * 2)).normalize(), 0}, // top
	{fvec3(0, 1, VIEWPORT_HEIGHT / (VIEWPORT_Z * 2)).normalize(), 0} // bottom
};

model getClippedModel(modelInstance const& mi, camera const& cam) {
	std::vector<fvec3> vertexes;
	vertexes.reserve(mi.m->vertexes.size());
	for (int i = 0; i < mi.m->vertexes.size(); i++) {
		fvec4 v = cam.transform * mi.transform * fvec4(mi.m->vertexes[i], 1);
		vertexes.push_back(fvec3(v.x, v.y, v.z));
	}

	std::vector<triangle> triangles;
	for (auto const& t : mi.m->triangles) {
		fvec3 n = crossProduct(
			vertexes[t.vs[1]] - vertexes[t.vs[0]],
			vertexes[t.vs[2]] - vertexes[t.vs[0]]
		);
		if (dotProduct(n, vertexes[t.vs[0]]) < 0) {
			triangle newTriangle = t;
			for (int i = 0; i < 3; i++) {
				newTriangle.normals[i] = (fvec3)((cam.rotation * mi.rotation) * fvec4(t.normals[i], 0));
			}
			triangles.push_back(newTriangle);
		}
	}
	for (auto const& plane : cPlanes) {
		std::vector<triangle> newTriangles;
		int sz = triangles.size();
		for (int i = 0; i < sz; i++) {
			std::vector<int> positive, negative;
			for (int j = 0; j < 3; j++) {
				if (plane.dst(vertexes[triangles[i].vs[j]]) >= 0) {
					positive.push_back(j);
				} else {
					negative.push_back(j);
				}
			}
			if (positive.size() == 3) {
				newTriangles.push_back(triangles[i]);
			} else if (positive.size() == 1) {
				fvec3 vp0 = vertexes[triangles[i].vs[positive[0]]];
				fvec3 vn0 = vertexes[triangles[i].vs[negative[0]]];
				fvec3 vn1 = vertexes[triangles[i].vs[negative[1]]];

				float intCoef00 = plane.intersectionCoef(vp0, vn0);
				float intCoef01 = plane.intersectionCoef(vp0, vn1);

				vertexes.push_back(plane.intersection(vp0, vn0, intCoef00));
				vertexes.push_back(plane.intersection(vp0, vn1, intCoef01));

				newTriangles.push_back(triangle(
					triangles[i].vs[positive[0]],
					vertexes.size() - 2, // vp0, vn0
					vertexes.size() - 1, // vp0, vn1
					triangles[i].color,
					triangles[i].normals[positive[0]],
					interpolatedVec(triangles[i].normals[positive[0]], triangles[i].normals[negative[0]], intCoef00),
					interpolatedVec(triangles[i].normals[positive[0]], triangles[i].normals[negative[1]], intCoef01),
					triangles[i].diffuse_map,
					triangles[i].uvs[positive[0]],
					interpolatedVec2(triangles[i].uvs[positive[0]], triangles[i].uvs[negative[0]], intCoef00),
					interpolatedVec2(triangles[i].uvs[positive[0]], triangles[i].uvs[negative[1]], intCoef01)
				));
			} else if (positive.size() == 2) {
				fvec3 vn0 = vertexes[triangles[i].vs[negative[0]]];
				fvec3 vp0 = vertexes[triangles[i].vs[positive[0]]];
				fvec3 vp1 = vertexes[triangles[i].vs[positive[1]]];

				vertexes.push_back(plane.intersection(vn0, vp0));
				vertexes.push_back(plane.intersection(vn0, vp1));

				float intCoef00 = plane.intersectionCoef(vn0, vp0);
				float intCoef01 = plane.intersectionCoef(vn0, vp1);

				fvec3 vn0vp0norm = interpolatedVec(triangles[i].normals[negative[0]], triangles[i].normals[positive[0]], intCoef00);
				fvec2 vn0vp0uv = interpolatedVec2(triangles[i].uvs[negative[0]], triangles[i].uvs[positive[0]], intCoef00);
				newTriangles.push_back(triangle(
					triangles[i].vs[positive[1]],
					vertexes.size() - 2, // vn0, vp0
					vertexes.size() - 1, // vn0, vp1
					triangles[i].color,
					triangles[i].normals[positive[1]],
					vn0vp0norm,
					interpolatedVec(triangles[i].normals[negative[0]], triangles[i].normals[positive[1]], intCoef01),
					triangles[i].diffuse_map,
					triangles[i].uvs[positive[1]],
					vn0vp0uv,
					interpolatedVec2(triangles[i].uvs[negative[0]], triangles[i].uvs[positive[1]], intCoef01)
				));
				newTriangles.push_back(triangle(
					triangles[i].vs[positive[1]],
					triangles[i].vs[positive[0]],
					vertexes.size() - 2, // vn0, vp0
					triangles[i].color,
					triangles[i].normals[positive[1]],
					triangles[i].normals[positive[0]],
					vn0vp0norm,
					triangles[i].diffuse_map,
					triangles[i].uvs[positive[1]],
					triangles[i].uvs[positive[0]],
					vn0vp0uv
				));
			}
		}
		triangles = newTriangles;
	}

	return model(vertexes, triangles);
}

void renderNormals(triangle const& t, model const& m, std::vector<ivec2> const& projected) {
	for (int i = 0; i < 3; i++) {
		ivec2 a = projectVertex(fvec4(m.vertexes[t.vs[i]], 1));
		ivec2 b = projectVertex(fvec4(t.normals[i] + m.vertexes[t.vs[i]], 1));
		drawLine(a, b, ivec3(0, 0, 0));
	}
}

void renderModelInstance(modelInstance const& mi, lightsContainer const& lights, camera const& cam) {
	model m = getClippedModel(mi, cam);
	std::vector<ivec2> projected;
	projected.reserve(mi.m->vertexes.size());
	for (auto& v : m.vertexes) {
		projected.push_back(projectVertex(fvec4(v, 1)));
	}
	for (auto& t : m.triangles) {
		renderTriangle(t, m, projected, lights, cam);
		//renderNormals(t, m, projected);
	}
}

void renderScene(camera const& cam, scene const& curScene) {
	lightsContainer lights = curScene.lights;
	for (directionLight& light : lights.directionLights) {
		light.direction = cam.rotation * fvec4(light.direction, 0);
	}
	for (pointLight& light : lights.pointLights) {
		light.pos = cam.transform * fvec4(light.pos, 0);
	}
	for (auto& mi : curScene.modelInstances) {
		renderModelInstance(mi, lights, cam);
	}
}

std::shared_ptr<model> generateSphere(int divs, ivec3 color) {
	std::vector<fvec3> vertexes;
	vertexes.reserve(divs * (divs + 1));
	for (int d = 0; d <= divs; d++) {
		// [0 .. divs] --> [-PI/2 .. PI/2] --> [-1 .. 1]
		float y = sin(d * M_PI / divs - M_PI / 2);
		float r = sqrt(1 - y * y);
		for (int i = 0; i < divs; i++) {
			float alpha = i * 2 * M_PI / divs;
			vertexes.emplace_back(r * cos(alpha), y, r * sin(alpha));
		}
	}

	std::vector<triangle> triangles;
	triangles.reserve(divs * divs * 2);
	for (int d = 0; d < divs; d++) {
		for (int i = 0; i < divs; i++) {
			int i1 = d * divs + i;
			int i2 = d * divs + (i + 1) % divs;
			triangles.emplace_back(i1, i2 + divs, i2, color, vertexes[i1], vertexes[i2 + divs], vertexes[i2], texture(), fvec2(0, 0), fvec2(0, 0), fvec2(0, 0));
			triangles.emplace_back(i1, i1 + divs, i2 + divs, color, vertexes[i1], vertexes[i1 + divs], vertexes[i2 + divs], texture(), fvec2(0, 0), fvec2(0, 0), fvec2(0, 0));
		}
	}

	return std::shared_ptr<model>(new model(vertexes, triangles));
}

std::shared_ptr<char> getBitmapData(Bitmap& bitmap) {
	BitmapData bitmapData;
	int w = bitmap.GetWidth();
	int h = bitmap.GetHeight();
	bitmap.LockBits(&Rect(0, 0, w, h), ImageLockModeRead, PixelFormat24bppRGB, &bitmapData);
	char* data = new char[w * h * 3];
	memcpy(data, bitmapData.Scan0, w * h * 3);
	return std::shared_ptr<char>(data, std::default_delete<char[]>());
}

scene createScene() {
	std::vector<fvec3> vertexes{
		fvec3(1, 1, 1),
		fvec3(-1, 1, 1),
		fvec3(-1, -1, 1),
		fvec3(1, -1, 1),
		fvec3(1, 1, -1),
		fvec3(-1, 1, -1),
		fvec3(-1, -1, -1),
		fvec3(1, -1, -1)
	};

	texture wood_texture(Bitmap(L"crate-texture.jpg"));

	std::vector<triangle> triangles{
		triangle(0, 1, 2, red, fvec3(0, 0, 1), fvec3(0, 0, 1), fvec3(0, 0, 1), wood_texture, fvec2(0, 0), fvec2(1, 0), fvec2(1, 1)),
		triangle(0, 2, 3, red, fvec3(0, 0, 1), fvec3(0, 0, 1), fvec3(0, 0, 1), wood_texture, fvec2(0, 0), fvec2(1, 1), fvec2(0, 1)),
		triangle(4, 0, 3, green, fvec3(1, 0, 0), fvec3(1, 0, 0), fvec3(1, 0, 0), wood_texture, fvec2(0, 0), fvec2(1, 0), fvec2(1, 1)),
		triangle(4, 3, 7, green, fvec3(1, 0, 0), fvec3(1, 0, 0), fvec3(1, 0, 0), wood_texture, fvec2(0, 0), fvec2(1, 1), fvec2(0, 1)),
		triangle(5, 4, 7, blue, fvec3(0, 0, -1), fvec3(0, 0, -1), fvec3(0, 0, -1), wood_texture, fvec2(0, 0), fvec2(1, 0), fvec2(1, 1)),
		triangle(5, 7, 6, blue, fvec3(0, 0, -1), fvec3(0, 0, -1), fvec3(0, 0, -1), wood_texture, fvec2(0, 0), fvec2(1, 1), fvec2(0, 1)),
		triangle(1, 5, 6, yellow, fvec3(-1, 0, 0), fvec3(-1, 0, 0), fvec3(-1, 0, 0), wood_texture, fvec2(0, 0), fvec2(1, 0), fvec2(1, 1)),
		triangle(1, 6, 2, yellow, fvec3(-1, 0, 0), fvec3(-1, 0, 0), fvec3(-1, 0, 0), wood_texture, fvec2(0, 0), fvec2(1, 1), fvec2(0, 1)),
		triangle(4, 5, 1, purple, fvec3(0, 1, 0), fvec3(0, 1, 0), fvec3(0, 1, 0), wood_texture, fvec2(0, 0), fvec2(1, 0), fvec2(1, 1)),
		triangle(4, 1, 0, purple, fvec3(0, 1, 0), fvec3(0, 1, 0), fvec3(0, 1, 0), wood_texture, fvec2(0, 0), fvec2(1, 1), fvec2(0, 1)),
		triangle(2, 6, 7, cyan, fvec3(0, -1, 0), fvec3(0, -1, 0), fvec3(0, -1, 0), wood_texture, fvec2(0, 0), fvec2(1, 0), fvec2(1, 1)),
		triangle(2, 7, 3, cyan, fvec3(0, -1, 0), fvec3(0, -1, 0), fvec3(0, -1, 0), wood_texture, fvec2(0, 0), fvec2(1, 1), fvec2(0, 1))
	};

	std::shared_ptr<model> cube(new model(vertexes, triangles));
	std::shared_ptr<model> sphere = generateSphere(20, green);
	std::shared_ptr<model> triangleModel(new model({ fvec3(0, 0, 0), fvec3(-3, 0, 0), fvec3(0, 1, 0) }, 
		{
			triangle(0, 1, 2, red, fvec3(0, 0, 1), fvec3(0, 0, 1), fvec3(0, 0, 1), wood_texture, fvec2(0, 0), fvec2(1, 0), fvec2(1, 1)) }));

	scene curScene;
	//curScene.modelInstances.emplace_back(triangleModel, 1, fvec3(0, 0, 0), fvec3(-1, 0, 5));
	curScene.modelInstances.emplace_back(cube, 0.75, fvec3(M_PI * 1.2, 0, M_PI / 4), fvec3(-2, 0, 6));
	curScene.modelInstances.emplace_back(cube, 1, fvec3(0, -195 * M_PI / 180, 0), fvec3(3.5, 3, 8));
	curScene.modelInstances.emplace_back(sphere, 0.8, fvec3(0, 0, 0), fvec3(-0.5, 0.5, 5.7));

	curScene.lights.directionLights.emplace_back(fvec3(-1, 0, 1), 0.2);
	curScene.lights.pointLights.emplace_back(fvec3(-3, 2, -10), 0.6);

	return curScene;
}

void updateScene(scene& curScene, int time) {
	curScene.modelInstances[0].rotate(fvec3(0.01, 0, 0));
	curScene.modelInstances[1].rotate(fvec3(0, -0.05, 0));
	curScene.modelInstances[2].rescale(1 + sin(time * 0.1) / 5);
}

void OnPaint(HDC hdc, scene& curScene, int time) {
	memset(data, -1, sizeof(data)); // fill white
	memset(depthBuf, 0, sizeof(depthBuf));

	updateScene(curScene, time);	

	renderScene(camera(fvec3(-3, 1, 0), fvec3(0, M_PI / 8, 0)), curScene);
		
	Graphics graphics(hdc);
	graphics.DrawImage(myImage, 0, 0);
}

void preproc() {
	BITMAPINFO bmi = { 0 };
	bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmi.bmiHeader.biWidth = WIDTH;
	bmi.bmiHeader.biHeight = HEIGHT;
	bmi.bmiHeader.biPlanes = 1;
	bmi.bmiHeader.biCompression = BI_RGB;
	bmi.bmiHeader.biBitCount = 24;
	myImage = new Bitmap(&bmi, data);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
	switch (message) {
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
}

INT WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, PSTR, INT iCmdShow) {
#ifdef _DEBUG
	AllocConsole();
	AttachConsole(GetCurrentProcessId());
	freopen("CON", "w", stdout);
#endif
	
	SetProcessDPIAware();
	HWND hWnd;
	WNDCLASS wndClass;
	GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;

	// Initialize GDI+.
	GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

	wndClass.style = CS_HREDRAW | CS_VREDRAW;
	wndClass.lpfnWndProc = WndProc;
	wndClass.cbClsExtra = 0;
	wndClass.cbWndExtra = 0;
	wndClass.hInstance = hInstance;
	wndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	wndClass.lpszMenuName = NULL;
	wndClass.lpszClassName = TEXT("Rasterizer");

	RegisterClass(&wndClass);

	RECT rect = { 0, 0, WIDTH, HEIGHT };
	AdjustWindowRectEx(&rect, WS_OVERLAPPEDWINDOW, false, WS_EX_OVERLAPPEDWINDOW);

	hWnd = CreateWindowEx(
		WS_EX_OVERLAPPEDWINDOW,
		TEXT("Rasterizer"),		// window class name
		TEXT("Rasterizer"),		// window caption
		WS_OVERLAPPEDWINDOW,	// window style
		CW_USEDEFAULT,			// initial x position
		CW_USEDEFAULT,			// initial y position
		rect.right - rect.left,	// initial x size
		rect.bottom - rect.top,	// initial y size
		NULL,					// parent window handle
		NULL,                   // window menu handle
		hInstance,              // program instance handle
		NULL);                  // creation parameters

	ShowWindow(hWnd, iCmdShow);
	UpdateWindow(hWnd);

	preproc();
	scene curScene = createScene();

	int time = 0;
	MSG msg = { 0 };
	while (msg.message != WM_QUIT) {
		if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		} else {
			HDC hdc;
			PAINTSTRUCT ps;
			hdc = GetDC(hWnd);
			OnPaint(hdc, curScene, time++);
			ReleaseDC(hWnd, hdc);
		}
	}

	GdiplusShutdown(gdiplusToken);
	return msg.wParam;
}