#define _USE_MATH_DEFINES

#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModelH.h"
#include <stdint.h>

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;
using glm::ivec2;
using glm::vec2;

#define SCREEN_WIDTH 320
#define SCREEN_HEIGHT 320
#define FULLSCREEN_MODE false

/*
 * STRUCTS
 */

struct Camera {
    float focalLength;
    vec4 position;
    mat4 R;
    float yaw;
};
Camera camera;

struct Pixel {
    int x;
    int y;
    float zinv;
    vec4 pos3d;
};

struct Vertex {
    vec4 position;
};

struct Light{
    vec4 pos;
    vec3 power;
    vec3 indirectLightPowerPerArea;
};
Light light;

float depthBuffer[SCREEN_WIDTH][SCREEN_HEIGHT];

mat4 identityMatrix = mat4(1.0f);

const float rotationIncr = 0.1;
const float translationIncr = 0.5;

vec4 currentNormal;
vec3 currentReflectance;
vec3 currentColour;

screen* sdlScreen;

/*
 * FUNCTIONS
 */

void Update();

void Draw(vector<Triangle>& triangles);

void Rotate();

void InitialiseStructs();

void InitialiseDepthBuffer();

void VertexShader(const Vertex& v, Pixel& p);

void PixelShader(Pixel& p, vec3 color);

void Interpolate(Pixel a, Pixel b, vector<Pixel>& result);

void DrawLineSDL(Pixel a, Pixel b, vec3 color);

void ComputePolygonRows(const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels);

void DrawPolygonRows(const vector<Pixel>& leftPixels, const vector<Pixel> rightPixels, vec3 color);

void DrawPolygon(const vector<Vertex>& vertices);

float CalcDist(vec4 a, vec4 b);

int main(int argc, char *argv[]) {
    sdlScreen = InitializeSDL(SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN_MODE);

    InitialiseStructs();

    vector<Triangle> triangles;
    LoadTestModel(triangles);

    while (NoQuitMessageSDL()) {
        Update();
        Draw(triangles);
        SDL_Renderframe(sdlScreen);
    }

    SDL_SaveImage(sdlScreen, "screenshot.bmp");

    KillSDL(sdlScreen);
    return 0;
}

void Draw(vector<Triangle>&triangles) {
    /* Clear buffer */
    memset(sdlScreen->buffer, 0, sdlScreen->height * sdlScreen->width * sizeof(uint32_t));
    InitialiseDepthBuffer();

    for (uint32_t i = 0; i<triangles.size(); ++i) {
        vector<Vertex> vertices(3);
        vertices[0].position = triangles[i].v0;
        vertices[1].position = triangles[i].v1;
        vertices[2].position = triangles[i].v2;
        currentNormal = triangles[i].normal;
//        currentColour = triangles[i].color;
//        currentReflectance = vec3(0.9, 0.9, 0.9);
        currentReflectance = triangles[i].color;

        DrawPolygon(vertices);
    }

}

void Update() {
    static int t = SDL_GetTicks();
    /* Compute frame time */
    int t2 = SDL_GetTicks();
    float dt = float(t2 - t);
    t = t2;

    std::cout << "Render time: " << dt << " ms." << std::endl;

    /* Update view */
    vec4 right(camera.R[0][0], camera.R[0][1], camera.R[0][2], 1);
    vec4 up(camera.R[1][0], camera.R[1][1], camera.R[1][2], 1);
    vec4 forward(camera.R[2][0], camera.R[2][1], camera.R[2][2], 1);

    const Uint8 *keyPress = SDL_GetKeyboardState(0);
    // Camera position
    if (keyPress[SDL_SCANCODE_UP]) {
        camera.position += translationIncr * up;
    }
    if (keyPress[SDL_SCANCODE_DOWN]) {
        camera.position -= translationIncr * up;
    }
    if (keyPress[SDL_SCANCODE_O]) {
        camera.position += translationIncr * forward;
    }
    if (keyPress[SDL_SCANCODE_K]) {
        camera.position -= translationIncr * forward;
    }
    if (keyPress[SDL_SCANCODE_RIGHT]) {
        camera.position += translationIncr * right;
    }
    if (keyPress[SDL_SCANCODE_LEFT]) {
        camera.position -= translationIncr * right;
    }
    // Rotate left
    if (keyPress[SDL_SCANCODE_Q]) {
        camera.position -= translationIncr * right;
        camera.yaw -= rotationIncr;
        Rotate();

    }
    // Rotate right
    if (keyPress[SDL_SCANCODE_E]) {
        camera.yaw += rotationIncr;
        Rotate();
        camera.position += translationIncr * right;
    }
    // Reset
    if (keyPress[SDL_SCANCODE_V]) {
        camera.position = vec4(0, 0, -3.001, 1);
        camera.yaw = 0;
        camera.R = identityMatrix;
        light.pos = vec4(0, -0.5, -0.7, 1);
    }
    // Light position
    if (keyPress[SDL_SCANCODE_W]) {
        light.pos.z += 0.1;
    }
    if (keyPress[SDL_SCANCODE_S]) {
        light.pos.z -= 0.1;
    }
    if (keyPress[SDL_SCANCODE_D]) {
        light.pos.x += 0.1;
    }
    if (keyPress[SDL_SCANCODE_A]) {
        light.pos.x -= 0.1;
    }
    if (keyPress[SDL_SCANCODE_R]) {
        light.pos.y += 0.1;
    }
    if (keyPress[SDL_SCANCODE_F]) {
        light.pos.y -= 0.1;
    }

}

void Rotate() {
    camera.R[0][0] = cos(camera.yaw);
    camera.R[0][2] = sin(camera.yaw);
    camera.R[2][0] = -sin(camera.yaw);
    camera.R[2][2] = cos(camera.yaw);
}

void InitialiseStructs() {
    /* Camera */
    camera.position = vec4(0, 0, -3.001, 1);
    camera.focalLength = SCREEN_HEIGHT;
    camera.yaw = 0;
    camera.R = identityMatrix;

    /* Lights */
    light.pos = vec4(0, -0.5, -0.7, 1);
    light.power = 34.f * vec3(1, 1, 1);
    light.indirectLightPowerPerArea = 0.5f * vec3(1, 1, 1);
}

void InitialiseDepthBuffer() {
    for(int i=0; i < SCREEN_WIDTH; i++)
        for(int j=0; j < SCREEN_HEIGHT; j++)
            depthBuffer[i][j] = -numeric_limits<int>::max();
}

void VertexShader(const Vertex& v, Pixel& pix) {
    Vertex point;
    point.position = (v.position - camera.position) * camera.R;

    pix.x = (int) (camera.focalLength * (point.position.x / point.position.z)) + (SCREEN_WIDTH / 2);
    pix.y = (int) (camera.focalLength * (point.position.y / point.position.z)) + (SCREEN_HEIGHT / 2);

    pix.zinv = 1.f/point.position.z;
    pix.pos3d = v.position;
}

void PixelShader(Pixel& p) {
    int x = p.x;
    int y = p.y;

//    if (!((x < 0) || (y < 0) || (x > SCREEN_WIDTH) || (y > SCREEN_HEIGHT))) {
        if (p.zinv > depthBuffer[x][y]) {
            p.pos3d.x /= p.zinv;
            p.pos3d.y /= p.zinv;
            p.pos3d.z /= p.zinv;

            vec4 r = glm::normalize(light.pos - p.pos3d);
            float dist = glm::length(light.pos - p.pos3d);
            float area = 4.f * M_PI * (dist * dist);
            float dot = glm::dot(glm::normalize(currentNormal), r);
            float max = glm::max(dot, 0.f);
            float div = max / area;

            vec3 directLight = light.power * div;
            vec3 illumination = (directLight + light.indirectLightPowerPerArea) * currentReflectance;

            PutPixelSDL(sdlScreen, x, y, illumination);
            depthBuffer[x][y] = p.zinv;
        }
//    }
}

void Interpolate(Pixel a, Pixel b, vector<Pixel>& result) {
    int N = result.size();

    vec3 current(a.x, a.y, a.zinv);
    vec4 currentPos = a.pos3d * a.zinv;

    float div = float(glm::max(N-1,1));
    vec3 step;
    step.x = (b.x - a.x) / div;
    step.y = (b.y - a.y) / div;
    step.z = (b.zinv - a.zinv) / div;
    vec4 posStep = (b.pos3d * b.zinv - a.pos3d * a.zinv) / div;

    for (int i = 0; i < N; i++) {
        Pixel pix;
        pix.x = current.x;
        pix.y = current.y;
        pix.zinv = current.z;
        pix.pos3d = currentPos/current.z;

        result[i] = pix;
        current += step;
        currentPos += posStep;
    }
}

void DrawLineSDL(Pixel a, Pixel b) {
    int delta_x = abs(a.x - b.x);
    int delta_y = abs(a.y - b.y);
    int pixels = glm::max(delta_x, delta_y) + 1;
    vector<Pixel> line(pixels);
    Interpolate(a, b, line);

    for (int i = 0; i < pixels; i++) {
        PixelShader(line[i]);

    }
}

void ComputePolygonRows(const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels) {
    int min_y = +numeric_limits<int>::max();
    int max_y = -numeric_limits<int>::max();

    int vPSize = vertexPixels.size();

    for (int i = 0; i < vPSize; i++) {
        if (vertexPixels[i].y > max_y) {
            max_y = vertexPixels[i].y;
        }
        if (vertexPixels[i].y < min_y) {
            min_y = vertexPixels[i].y;
        }
    }

    int rows = (max_y - min_y) + 1;
    rightPixels.resize(rows);
    leftPixels.resize(rows);

    for (int i = 0; i < rows; i++) {
        leftPixels[i].x = +numeric_limits<int>::max();
        rightPixels[i].x = -numeric_limits<int>::max();
        rightPixels[i].y = leftPixels[i].y = min_y + i;
    }

    for (int i = 0; i < vPSize; i++) {
        int j = (i + 1) % vPSize;
        int numPix = abs(vertexPixels[i].y - vertexPixels[j].y) + 1;
        vector<Pixel> line(numPix);
        Interpolate(vertexPixels[i], vertexPixels[j], line);

        for (int k = 0; k < numPix; k++) {
            int x = line[k].x;
            int y_ = line[k].y - min_y;
            if (y_ >= 0 && y_ < leftPixels.size()) {
                if (x > rightPixels[y_].x) {
                    rightPixels[y_] = line[k];
                }
                if (x < leftPixels[y_].x) {
                    leftPixels[y_] = line[k];
                };
            }
        }
    }
}

void DrawPolygonRows(const vector<Pixel>& leftPixels, const vector<Pixel> rightPixels) {
    int numPix = leftPixels.size();

    for (int i = 0; i < numPix; i++) {
        if (leftPixels[i].y >= 0 && leftPixels[i].y < SCREEN_HEIGHT) DrawLineSDL(leftPixels[i], rightPixels[i]);
    }
}

void DrawPolygon(const vector<Vertex>& vertices) {
    int V = vertices.size();
    vector<Pixel> vertexPixels(V);

    for(int i=0; i<V; ++i) VertexShader(vertices[i], vertexPixels[i]);

    vector<Pixel> leftPixels;
    vector<Pixel> rightPixels;

    ComputePolygonRows(vertexPixels, leftPixels, rightPixels);
    DrawPolygonRows(leftPixels, rightPixels);
}

float CalcDist(vec4 a, vec4 b) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    return sqrt((dx * dx) + (dy * dy) + (dz * dz));
}