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
using glm::vec2;
using glm::clamp;

#define SCREEN_HEIGHT 500
#define SCREEN_WIDTH 500
#define FULLSCREEN_MODE false
#define CLIP 1

/*
 * STRUCTS & CONSTS
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
    vec4 cameraPosition;
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

float depthBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];

mat4 identityMatrix = mat4(1.0f);
const float rotationIncr = 0.1;
const float translationIncr = 0.5;
const float lightTranslationIncr = 0.1;
vec4 currentNormal;
vec3 currentReflectance;

screen* sdlScreen;

/* FXAA */

vec3 colourBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];
const vec2 invScreen(1.0f / SCREEN_HEIGHT, 1.0f / SCREEN_WIDTH);
const float edgeMin = 1.f / 8.f;
const float edgeMax = 1.f / 128.f;
const float gradScale = 2.5f;
const float fxaaConst = 1.f / 6.f;
const vec3 luma(0.299f, 0.587f, 0.114f);

/*
 * FUNCTIONS
 */

void Update();

void Draw(vector<Triangle>& triangles);

void Rotate();

void InitialiseStructs();

void InitialiseDepthBuffer();

void InitialiseColourBuffer();

void VertexShader(const Vertex& v, Pixel& p);

void PixelShader(Pixel& p, vec3 color);

void Interpolate(Pixel a, Pixel b, vector<Pixel>& result);

void DrawLineSDL(Pixel a, Pixel b, vec3 color);

void ComputePolygonRows(const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels);

void DrawPolygonRows(const vector<Pixel>& leftPixels, const vector<Pixel> rightPixels, vec3 color);

void DrawPolygon(const vector<Vertex>& vertices);

vec3 GetFromBuffer(vec2 point);

vec3 GetFromBuffer(float x, float y);

vec3 FXAA(int x, int y);

void ClipPolygon(vector<Pixel>& vertexPixels);

void ClipNearest(vector<Pixel>& vertexPixels);

void ClipEdge(vector<Pixel>& vertexPixels, vec2 edgeVertex[]);

bool OnEdgeVertex(Pixel pixel, vec2 edgeVertex[]);

void Intersection(Pixel pixel1, Pixel pixel2, vec2 edgeVertex[], Pixel& intersection);

int main(int argc, char *argv[]) {
    sdlScreen = InitializeSDL(SCREEN_HEIGHT, SCREEN_WIDTH, FULLSCREEN_MODE);

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
    InitialiseColourBuffer();

    for (int i = 0; i < triangles.size(); ++i) {
        vector<Vertex> vertices(3);
        vertices[0].position = triangles[i].v0;
        vertices[1].position = triangles[i].v1;
        vertices[2].position = triangles[i].v2;
        currentNormal = triangles[i].normal;
        currentReflectance = triangles[i].color;

        /* Backface culling */
        vec4 viewVector = vertices[0].position * camera.R - camera.position;
        bool backface = glm::dot(currentNormal, viewVector) < 0;

        if (backface) DrawPolygon(vertices);
    }

    for (int i = 0; i < SCREEN_HEIGHT; i++) {
        for (int j = 0; j < SCREEN_WIDTH; j++) {
            PutPixelSDL(sdlScreen, i, j, FXAA(i, j));
        }
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

    const Uint8 *keyState = SDL_GetKeyboardState(0);
    /* Camera position */
    if (keyState[SDL_SCANCODE_UP]) {
        camera.position += translationIncr * up;
    }
    if (keyState[SDL_SCANCODE_DOWN]) {
        camera.position -= translationIncr * up;
    }
    if (keyState[SDL_SCANCODE_O]) {
        camera.position += 0.1f * forward;
    }
    if (keyState[SDL_SCANCODE_K]) {
        camera.position -= 0.1f * forward;
    }
    if (keyState[SDL_SCANCODE_RIGHT]) {
        camera.position += translationIncr * right;
    }
    if (keyState[SDL_SCANCODE_LEFT]) {
        camera.position -= translationIncr * right;
    }
    /* Rotate left */
    if (keyState[SDL_SCANCODE_Q]) {
        camera.yaw -= rotationIncr;
        Rotate();
    }
    /* Rotate right */
    if (keyState[SDL_SCANCODE_E]) {
        camera.yaw += rotationIncr;
        Rotate();
    }
    /* Light position */
    if (keyState[SDL_SCANCODE_W]) {
        light.pos.z += lightTranslationIncr;
    }
    if (keyState[SDL_SCANCODE_S]) {
        light.pos.z -= lightTranslationIncr;
    }
    if (keyState[SDL_SCANCODE_D]) {
        light.pos.x += lightTranslationIncr;
    }
    if (keyState[SDL_SCANCODE_A]) {
        light.pos.x -= lightTranslationIncr;
    }
    if (keyState[SDL_SCANCODE_R]) {
        light.pos.y -= lightTranslationIncr;
    }
    if (keyState[SDL_SCANCODE_F]) {
        light.pos.y += lightTranslationIncr;
    }
    /* Reset */
    if (keyState[SDL_SCANCODE_V]) {
        camera.position = vec4(0, 0, -3.001, 1);
        camera.yaw = 0;
        camera.R = identityMatrix;
        light.pos = vec4(0, -0.5, -0.7, 1);
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
    light.power = 14.f * vec3(1, 1, 1);
    light.indirectLightPowerPerArea = 0.5f * vec3(1, 1, 1);
}

void InitialiseDepthBuffer() {
    for(int i = 0; i < SCREEN_HEIGHT; i++)
        for(int j = 0; j < SCREEN_WIDTH; j++) {
            depthBuffer[i][j] = -numeric_limits<int>::max();
        }
}

void InitialiseColourBuffer() {
    for(int i = 0; i < SCREEN_HEIGHT; i++)
        for(int j = 0; j < SCREEN_WIDTH; j++) {
            colourBuffer[i][j] = vec3(0.f, 0.f, 0.f);
        }
}

void VertexShader(const Vertex& v, Pixel& pix) {
    pix.cameraPosition = (v.position - camera.position) * camera.R;

    pix.x = (int) (camera.focalLength * (pix.cameraPosition.x / pix.cameraPosition.z)) + (SCREEN_WIDTH / 2);
    pix.y = (int) (camera.focalLength * (pix.cameraPosition.y / pix.cameraPosition.z)) + (SCREEN_HEIGHT / 2);

    pix.zinv = 1.f / pix.cameraPosition.z;
    pix.pos3d = v.position;
}

void PixelShader(Pixel& p) {
    int x = p.x;
    int y = p.y;

    if ((x >= 0) && (y >= 0) && (x < SCREEN_HEIGHT) && (y < SCREEN_WIDTH)) {
        if (p.zinv > depthBuffer[x][y]) {

            vec4 r = glm::normalize(light.pos - p.pos3d);
            float dist = glm::length(light.pos - p.pos3d);
            float area = 4.f * M_PI * (dist * dist);
            float dot = glm::dot(glm::normalize(currentNormal), r);
            float max_ = glm::max(dot, 0.f);
            float div = max_ / area;

            vec3 directLight = light.power * div;
            vec3 illumination = (directLight + light.indirectLightPowerPerArea) * currentReflectance;

            colourBuffer[x][y] = illumination;
            depthBuffer[x][y] = p.zinv;
        }
    }
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

    result[N-1].x = b.x;
    result[N-1].y = b.y;
    result[N-1].zinv = b.zinv;
    result[N-1].pos3d = b.pos3d;
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
        if (leftPixels[i].y >= 0 && leftPixels[i].y < SCREEN_WIDTH) DrawLineSDL(leftPixels[i], rightPixels[i]);
    }
}

void DrawPolygon(const vector<Vertex>& vertices) {
    int V = vertices.size();
    vector<Pixel> vertexPixels(V);

    for(int i=0; i<V; ++i) VertexShader(vertices[i], vertexPixels[i]);

    /* Polygon Clipping */
    if (CLIP == 1) ClipPolygon(vertexPixels);

    vector<Pixel> leftPixels;
    vector<Pixel> rightPixels;

    ComputePolygonRows(vertexPixels, leftPixels, rightPixels);
    DrawPolygonRows(leftPixels, rightPixels);
}

vec3 GetFromBuffer(vec2 point) {
    return GetFromBuffer(point.x, point.y);
}

vec3 GetFromBuffer(float x, float y) {
    int width_ = SCREEN_HEIGHT - 1;
    int height_ = SCREEN_WIDTH - 1;

    int x_ = clamp((int) (clamp(x, 0.f, 1.f) * width_), 0, width_);
    int y_ = clamp((int) (clamp(y, 0.f, 1.f) * height_), 0, height_);

    return colourBuffer[x_][y_];
}

vec3 FXAA(int x, int y) {
    vec2 point(((float) x / (float) SCREEN_HEIGHT), ((float) y / (float) SCREEN_WIDTH));

    vec3 centre = GetFromBuffer(point);
    vec3 NW = GetFromBuffer(point.x - invScreen.x, point.y - invScreen.y);
    vec3 NE = GetFromBuffer(point.x + invScreen.x, point.y - invScreen.y);
    vec3 SW = GetFromBuffer(point.x - invScreen.x, point.y + invScreen.y);
    vec3 SE = GetFromBuffer(point.x + invScreen.x, point.y + invScreen.y);

    float lumaCentre = glm::dot(centre, luma);
    float lumaNW = glm::dot(NW, luma);
    float lumaNE = glm::dot(NE, luma);
    float lumaSW = glm::dot(SW, luma);
    float lumaSE = glm::dot(SE, luma);

    float lumaMin = glm::min(lumaCentre, glm::min(lumaNW, glm::min(lumaNE, glm::min(lumaSW, lumaSE))));
    float lumaMax = glm::max(lumaCentre, glm::max(lumaNW, glm::max(lumaNE, glm::max(lumaSW, lumaSE))));

    float lumaNorthEdge = lumaNW + lumaNE;
    float lumaSouthEdge = lumaSW + lumaSE;
    float lumaWestEdge = lumaNW + lumaSW;
    float lumaEastEdge = lumaNE + lumaSE;

    vec2 grad(-(lumaNorthEdge - lumaSouthEdge), (lumaWestEdge - lumaEastEdge));
    float temp = (lumaNW + lumaNE + lumaSW + lumaSE) * edgeMax * 0.25;
    float gradThreshold = glm::max(temp, edgeMin);
    float gradMin = 1.f / (glm::min(abs(grad.x), abs(grad.y)) + gradThreshold);

    float gradX = glm::min(gradScale, glm::max(-gradScale, grad.x * gradMin)) * invScreen.x;
    float gradY = glm::min(gradScale, glm::max(-gradScale, grad.y * gradMin)) * invScreen.y;

    grad = vec2(gradX, gradY);

    vec3 A = 0.5f * (GetFromBuffer(point + (-fxaaConst * grad)) + GetFromBuffer(point + (fxaaConst * grad)));
    vec3 B = 0.5f * A + 0.25f * (GetFromBuffer(point + (-0.5f * grad)) + GetFromBuffer(point + (0.5f * grad)));

    float lumaB = glm::dot(B, luma);

    if (lumaB < lumaMin || lumaB > lumaMax) {
        return A;
    } else {
        return B;
    }
}

void ClipPolygon(vector<Pixel>& vertexPixels) {
    vec2 edgeVertices[2];

    ClipNearest(vertexPixels);

    /* Top edge */
    edgeVertices[0] = vec2(SCREEN_HEIGHT, 0);
    edgeVertices[1] = vec2(0, 0);
    ClipEdge(vertexPixels, edgeVertices);
    /* Left edge */
    edgeVertices[0] = vec2(0, 0);
    edgeVertices[1] = vec2(0, SCREEN_WIDTH);
    ClipEdge(vertexPixels, edgeVertices);
    /* Right edge */
    edgeVertices[0] = vec2(SCREEN_HEIGHT, SCREEN_WIDTH);
    edgeVertices[1] = vec2(SCREEN_WIDTH, 0);
    ClipEdge(vertexPixels, edgeVertices);
    /* Bottom edge */
    edgeVertices[0] = vec2(0, SCREEN_WIDTH);
    edgeVertices[1] = vec2(SCREEN_HEIGHT, SCREEN_WIDTH);
    ClipEdge(vertexPixels, edgeVertices);

}

void ClipNearest(vector<Pixel>& vertexPixels) {
    int vSize = vertexPixels.size();

    vector<Pixel> clipped;
    Pixel pixel1 = vertexPixels[vSize - 1];
    Pixel pixel2;

    for (int i = 0; i < vSize; i++) {
        pixel2 = vertexPixels[i];

        if ((pixel1.cameraPosition.z >= 0.0001) && (pixel2.cameraPosition.z >= 0.0001)) {
            clipped.push_back(pixel2);
        } else if ((pixel1.cameraPosition.z >= 0.0001) && !(pixel2.cameraPosition.z >= 0.0001)) {
            float diff = (pixel1.cameraPosition.z - 0.0001) / (pixel1.cameraPosition.z - pixel2.cameraPosition.z);
            Pixel intersection;
            intersection.cameraPosition = ((1 - diff) * pixel1.cameraPosition) + (diff * pixel2.cameraPosition);
            intersection.pos3d = ((1 - diff) * pixel1.pos3d) + (diff * pixel2.pos3d);
            intersection.x = (int) (camera.focalLength * (intersection.cameraPosition.x / intersection.cameraPosition.z)) + (SCREEN_HEIGHT / 2);
            intersection.y = (int) (camera.focalLength * (intersection.cameraPosition.y / intersection.cameraPosition.z)) + (SCREEN_WIDTH / 2);
            intersection.zinv = 1.f / intersection.cameraPosition.z;

            clipped.push_back(intersection);
        } else if (!(pixel1.cameraPosition.z >= 0.0001) && (pixel2.cameraPosition.z >= 0.0001)) {
            float diff = (pixel1.cameraPosition.z - 0.0001) / (pixel1.cameraPosition.z - pixel2.cameraPosition.z);
            Pixel intersection;
            intersection.cameraPosition = ((1 - diff) * pixel1.cameraPosition) + (diff * pixel2.cameraPosition);
            intersection.pos3d = ((1 - diff) * pixel1.pos3d) + (diff * pixel2.pos3d);
            intersection.x = (int) (camera.focalLength * (intersection.cameraPosition.x / intersection.cameraPosition.z)) + (SCREEN_HEIGHT / 2);
            intersection.y = (int) (camera.focalLength * (intersection.cameraPosition.y / intersection.cameraPosition.z)) + (SCREEN_WIDTH / 2);
            intersection.zinv = 1.f / intersection.cameraPosition.z;

            clipped.push_back(intersection);
            clipped.push_back(pixel2);
        }

        pixel1 = pixel2;
    }

    vertexPixels = clipped;
}

void ClipEdge(vector<Pixel>& vertexPixels, vec2 edgeVertex[]) {
    int vSize = vertexPixels.size();

    vector<Pixel> clipped;
    Pixel pixel1 = vertexPixels[vSize - 1];
    Pixel pixel2;

    for (int i = 0; i < vSize; i++) {
        pixel2 = vertexPixels[i];

        if ((OnEdgeVertex(pixel1, edgeVertex)) && (OnEdgeVertex(pixel2, edgeVertex))) {
            clipped.push_back(pixel2);
        } else if (((OnEdgeVertex(pixel1, edgeVertex)) && !(OnEdgeVertex(pixel2, edgeVertex)))) {
            Pixel intersection;
            Intersection(pixel1, pixel2, edgeVertex, intersection);

            clipped.push_back(intersection);
        } else if (!(OnEdgeVertex(pixel1, edgeVertex)) && (OnEdgeVertex(pixel2, edgeVertex))) {
            Pixel intersection;
            Intersection(pixel1, pixel2, edgeVertex, intersection);

            clipped.push_back(intersection);
            clipped.push_back(pixel2);
        }

        pixel1 = pixel2;
    }

    vertexPixels = clipped;
}

bool OnEdgeVertex(Pixel pixel, vec2 edgeVertex[]) {
    /* Top edge */
    if ((edgeVertex[1].x < edgeVertex[0].x) && (pixel.y >= edgeVertex[0].y)) {
        return true;
    }
    /* Left edge */
    else if ((edgeVertex[1].y > edgeVertex[0].y) && (pixel.x >= edgeVertex[0].x)) {
        return true;
    }
    /* Right edge */
    else if ((edgeVertex[1].y < edgeVertex[0].y) && (pixel.x <= edgeVertex[0].x)) {
        return true;
    }
    /* Bottom edge */
    else if ((edgeVertex[1].x > edgeVertex[0].x) && (pixel.y <= edgeVertex[0].y)) {
        return true;
    } else {
        return false;
    }
}

void Intersection(Pixel pixel1, Pixel pixel2, vec2 edgeVertex[], Pixel& intersection) {
    /* Top edge */
    if (edgeVertex[1].x < edgeVertex[0].x) {
        float diff = (2 * pixel1.cameraPosition.y + pixel1.cameraPosition.z) /
                     ((2 * pixel1.cameraPosition.y + pixel1.cameraPosition.z) - (2 * pixel2.cameraPosition.y + pixel2.cameraPosition.z));
        intersection.cameraPosition = ((1 - diff) * pixel1.cameraPosition) + (diff * pixel2.cameraPosition);
        intersection.pos3d = ((1 - diff) * pixel1.pos3d) + (diff * pixel2.pos3d);
    }
    /* Bottom edge*/
    else if (edgeVertex[1].x > edgeVertex[0].x) {
        float diff = (2 * pixel1.cameraPosition.y - pixel1.cameraPosition.z) /
                ((2 * pixel1.cameraPosition.y - pixel1.cameraPosition.z) - (2 * pixel2.cameraPosition.y - pixel2.cameraPosition.z));
        intersection.cameraPosition = ((1 - diff) * pixel1.cameraPosition) + (diff * pixel2.cameraPosition);
        intersection.pos3d = ((1 - diff) * pixel1.pos3d) + (diff * pixel2.pos3d);
    }
    /* Left edge */
    else if (edgeVertex[1].y > edgeVertex[0].y) {
        float diff = (2 * pixel1.cameraPosition.x + pixel1.cameraPosition.z) /
                     ((2 * pixel1.cameraPosition.x + pixel1.cameraPosition.z) - (2 * pixel2.cameraPosition.x + pixel2.cameraPosition.z));
        intersection.cameraPosition = ((1 - diff) * pixel1.cameraPosition) + (diff * pixel2.cameraPosition);
        intersection.pos3d = ((1 - diff) * pixel1.pos3d) + (diff * pixel2.pos3d);
    }
    /* Right edge*/
    else if (edgeVertex[1].y < edgeVertex[0].y) {
        float diff = (2 * pixel1.cameraPosition.x - pixel1.cameraPosition.z) /
                     ((2 * pixel1.cameraPosition.x - pixel1.cameraPosition.z) - (2 * pixel2.cameraPosition.x - pixel2.cameraPosition.z));
        intersection.cameraPosition = ((1 - diff) * pixel1.cameraPosition) + (diff * pixel2.cameraPosition);
        intersection.pos3d = ((1 - diff) * pixel1.pos3d) + (diff * pixel2.pos3d);
    }

    intersection.x = (int) (camera.focalLength * (intersection.cameraPosition.x / intersection.cameraPosition.z)) + (SCREEN_HEIGHT / 2);
    intersection.y = (int) (camera.focalLength * (intersection.cameraPosition.y / intersection.cameraPosition.z)) + (SCREEN_WIDTH / 2);
    intersection.zinv = 1.f / intersection.cameraPosition.z;
}
