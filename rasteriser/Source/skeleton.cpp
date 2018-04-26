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
#define CLIP true
#define SHADOWS true

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
    int x, y;
    float zinv;
    vec4 pos3d;
    vec4 cameraPosition;
    float isLit = 1.f;
};

struct Vertex {
    vec4 position;
};

struct Light{
    vec4 pos;
    vec3 power;
    vec3 indirectLightPowerPerArea;
    float focalLength;
};
Light light;

float depthBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];
float lightBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];

mat4 identityMatrix = mat4(1.0f);
const float rotationIncr = 2 * (M_PI / 180);
const float translationIncr = 0.1f;
const float lightTranslationIncr = 0.1f;
vec4 currentNormal;
vec3 currentReflectance;

screen* sdlScreen;

/* FXAA */

vec3 colourBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];
const vec2 invScreen(1.0f / SCREEN_WIDTH, 1.0f / SCREEN_HEIGHT);
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

void Rotate(Camera& camera_);

void InitialiseStructs();

void InitialiseBuffers();

void VertexShader(const Vertex& v, Pixel& p);

void PixelShader(Pixel& p);

void Interpolate(Pixel a, Pixel b, vector<Pixel>& result);

void ComputePolygonRows(const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels);

void DrawPolygonRows(const vector<Pixel>& leftPixels, const vector<Pixel> rightPixels);

void DrawPolygon(const vector<Vertex>& vertices);

vec3 GetFromBuffer(vec2 point);

vec3 GetFromBuffer(float x, float y);

vec3 FXAA(int x, int y);

void ClipPolygon(vector<Pixel>& vertexPixels);

void ClipZPlane(vector<Pixel>& vertexPixels);

void ClipEdges(vector<Pixel>& vertexPixels, vec2 edgeBounds[]);

bool OnEdge(Pixel pixel, vec2 edgeBounds[]);

void Intermediate(Pixel pixel1, Pixel pixel2, vec2 edgeBounds[], Pixel& intermediate);

void DrawShadows(const vector<Vertex>& vertices);

void ShadowRows(const vector<Pixel>& leftPixels, const vector<Pixel> rightPixels);

void LightVertexShader(const Vertex& v, Pixel& p);

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
    InitialiseBuffers();

    if (SHADOWS) {
        for (int i = 0; i < triangles.size(); ++i) {
            vector<Vertex> vertices(3);
            vertices[0].position = triangles[i].v0;
            vertices[1].position = triangles[i].v1;
            vertices[2].position = triangles[i].v2;
            currentNormal = triangles[i].normal;
            currentReflectance = triangles[i].color;

            DrawShadows(vertices);
        }
    }

    for (int i = 0; i < triangles.size(); ++i) {
        vector<Vertex> vertices(3);
        vertices[0].position = triangles[i].v0;
        vertices[1].position = triangles[i].v1;
        vertices[2].position = triangles[i].v2;
        currentNormal = triangles[i].normal;
        currentReflectance = triangles[i].color;

        DrawPolygon(vertices);
    }

    for (int y = 0; y < SCREEN_HEIGHT; y++) {
        for (int x = 0; x < SCREEN_WIDTH; x++) {
            PutPixelSDL(sdlScreen, x, y, FXAA(x, y));
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
        camera.position += translationIncr * forward;
    }
    if (keyState[SDL_SCANCODE_K]) {
        camera.position -= translationIncr * forward;
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
        Rotate(camera);
    }
    /* Rotate right */
    if (keyState[SDL_SCANCODE_E]) {
        camera.yaw += rotationIncr;
        Rotate(camera);
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
        InitialiseStructs();
    }
}

void Rotate(Camera& camera_) {
    camera.R[0][0] = cos(camera_.yaw);
    camera.R[0][2] = sin(camera_.yaw);
    camera.R[2][0] = -sin(camera_.yaw);
    camera.R[2][2] = cos(camera_.yaw);
}

void InitialiseStructs() {
    /* Camera */
    camera.position = vec4(0, 0, -3.001, 1);
    camera.focalLength = SCREEN_HEIGHT;
    camera.yaw = 0;
    camera.R = identityMatrix;

    /* Lights */
    if(SHADOWS) {
        light.pos = vec4(0.5f, 0.5f, -4.f, 1.f);
        light.power = 100.f * vec3(1, 1, 1);
    } else {
        light.pos = vec4(0.f, -0.5f, -0.7f, 1.f);
        light.power = 11.f * vec3(1, 1, 1);
    }
    light.indirectLightPowerPerArea = 0.5f * vec3(1, 1, 1);
    light.focalLength = SCREEN_HEIGHT;
}

void InitialiseBuffers() {
    for(int y = 0; y < SCREEN_HEIGHT; y++)
        for(int x = 0; x < SCREEN_WIDTH; x++) {
            depthBuffer[y][x] = -numeric_limits<int>::max();
            lightBuffer[y][x] = -numeric_limits<int>::max();
            colourBuffer[y][x] = vec3(0.f, 0.f, 0.f);
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
    float zinv = p.zinv;

    if (SHADOWS) {
        vec4 lightPosition = (p.pos3d - light.pos);
        int lx = (int) (camera.focalLength * (lightPosition.x / lightPosition.z)) + (SCREEN_WIDTH / 2);
        int ly = (int) (camera.focalLength * (lightPosition.y / lightPosition.z)) + (SCREEN_HEIGHT / 2);
        float lzinv = 1.f / lightPosition.z;
        if ((lzinv + 0.01f) <= lightBuffer[ly][lx]) {
            p.isLit = 0.f;
        }
    }

    vec3 illumination = light.indirectLightPowerPerArea * currentReflectance;

    if (zinv >= depthBuffer[y][x]) {

        if (p.isLit == 1.f) {
            vec4 r = glm::normalize(light.pos - p.pos3d);
            float dist = glm::length(light.pos - p.pos3d);
            float invArea = 1.f / (4.f * M_PI * (dist * dist));
            float dot = glm::dot(glm::normalize(currentNormal), r);
            float max_ = glm::max(dot, 0.f);
            float div = max_ * invArea;

            vec3 directLight = light.power * div;
            illumination = (directLight + light.indirectLightPowerPerArea) * currentReflectance;
        }

        colourBuffer[y][x] = illumination;
        depthBuffer[y][x] = zinv;
    }
}

void Interpolate(Pixel a, Pixel b, vector<Pixel>& result) {
    int N = result.size();

    vec3 current(a.x, a.y, a.zinv);
    vec4 currentPos = a.pos3d * a.zinv;

    float div = 1.f / float(glm::max(N - 1, 1));
    vec3 step;
    step.x = (b.x - a.x) * div;
    step.y = (b.y - a.y) * div;
    step.z = (b.zinv - a.zinv) * div;
    vec4 posStep = (b.pos3d * b.zinv - a.pos3d * a.zinv) * div;

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
        if (leftPixels[i].y >= 0 && leftPixels[i].y < SCREEN_WIDTH) {
            Pixel a = leftPixels[i];
            Pixel b = rightPixels[i];
            int delta_x = abs(a.x - b.x);
            int delta_y = abs(a.y - b.y);
            int pixels = glm::max(delta_x, delta_y) + 1;
            vector<Pixel> line(pixels);
            Interpolate(a, b, line);
            for (int j = 0; j < pixels; j++) {
                Pixel pix = line[j];
                bool pixelInBox = ((pix.x >= 0) && (pix.x < SCREEN_WIDTH) && (pix.y >= 0)  && (pix.y < SCREEN_HEIGHT));
                if (pixelInBox) PixelShader(line[j]);
            }
        }
    }
}

void DrawPolygon(const vector<Vertex>& vertices) {
    int V = vertices.size();
    vector<Pixel> vertexPixels(V);

    for(int i=0; i<V; ++i) VertexShader(vertices[i], vertexPixels[i]);

    /* Polygon Clipping */
    if (CLIP) ClipPolygon(vertexPixels);

    vector<Pixel> leftPixels;
    vector<Pixel> rightPixels;

    ComputePolygonRows(vertexPixels, leftPixels, rightPixels); // with lx
    DrawPolygonRows(leftPixels, rightPixels);
}

vec3 GetFromBuffer(vec2 point) {
    return GetFromBuffer(point.x, point.y);
}

vec3 GetFromBuffer(float x, float y) {
    int width_ = SCREEN_WIDTH - 1;
    int height_ = SCREEN_HEIGHT - 1;

    int x_ = clamp((int) (clamp(x, 0.f, 1.f) * width_), 0, width_);
    int y_ = clamp((int) (clamp(y, 0.f, 1.f) * height_), 0, height_);

    return colourBuffer[y_][x_];
}

vec3 FXAA(int x, int y) {
    vec2 point(((float) x / (float) SCREEN_WIDTH), ((float) y / (float) SCREEN_HEIGHT));

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

    ClipZPlane(vertexPixels);

    /* Top edge */  // x, y
    edgeVertices[0] = vec2(SCREEN_WIDTH, 0);
    edgeVertices[1] = vec2(0, 0);
    ClipEdges(vertexPixels, edgeVertices);

    /* Left edge */
    edgeVertices[0] = vec2(0, 0);
    edgeVertices[1] = vec2(0, SCREEN_HEIGHT);
    ClipEdges(vertexPixels, edgeVertices);

    /* Right edge */
    edgeVertices[0] = vec2(SCREEN_WIDTH, SCREEN_HEIGHT);
    edgeVertices[1] = vec2(SCREEN_WIDTH, 0);
    ClipEdges(vertexPixels, edgeVertices);

    /* Bottom edge */
    edgeVertices[0] = vec2(0, SCREEN_HEIGHT);
    edgeVertices[1] = vec2(SCREEN_WIDTH, SCREEN_HEIGHT);
    ClipEdges(vertexPixels, edgeVertices);

}

void ClipZPlane(vector<Pixel>& vertexPixels) {
    int vSize = vertexPixels.size();

    vector<Pixel> clipped;
    Pixel pixel1 = vertexPixels[vSize - 1];
    Pixel pixel2;
    float nearClipThreshold = 0.0001f;

    for (int i = 0; i < vSize; i++) {
        pixel2 = vertexPixels[i];

        if ((pixel1.cameraPosition.z >= nearClipThreshold) && (pixel2.cameraPosition.z >= nearClipThreshold)) {
            clipped.push_back(pixel2);
        } else if ((pixel1.cameraPosition.z >= nearClipThreshold) && !(pixel2.cameraPosition.z >= nearClipThreshold)) {
            float diff = (pixel1.cameraPosition.z - nearClipThreshold) / (pixel1.cameraPosition.z - pixel2.cameraPosition.z);
            Pixel intermediate;
            intermediate.cameraPosition = ((1 - diff) * pixel1.cameraPosition) + (diff * pixel2.cameraPosition);
            intermediate.pos3d = ((1 - diff) * pixel1.pos3d) + (diff * pixel2.pos3d);
            intermediate.x = (int) (camera.focalLength * (intermediate.cameraPosition.x / intermediate.cameraPosition.z)) + (SCREEN_WIDTH / 2);
            intermediate.y = (int) (camera.focalLength * (intermediate.cameraPosition.y / intermediate.cameraPosition.z)) + (SCREEN_HEIGHT / 2);
            intermediate.zinv = 1.f / intermediate.cameraPosition.z;

            clipped.push_back(intermediate);
        } else if (!(pixel1.cameraPosition.z >= nearClipThreshold) && (pixel2.cameraPosition.z >= nearClipThreshold)) {
            float diff = (pixel1.cameraPosition.z - nearClipThreshold) / (pixel1.cameraPosition.z - pixel2.cameraPosition.z);
            Pixel intermediate;
            intermediate.cameraPosition = ((1 - diff) * pixel1.cameraPosition) + (diff * pixel2.cameraPosition);
            intermediate.pos3d = ((1 - diff) * pixel1.pos3d) + (diff * pixel2.pos3d);
            intermediate.x = (int) (camera.focalLength * (intermediate.cameraPosition.x / intermediate.cameraPosition.z)) + (SCREEN_WIDTH / 2);
            intermediate.y = (int) (camera.focalLength * (intermediate.cameraPosition.y / intermediate.cameraPosition.z)) + (SCREEN_HEIGHT / 2);
            intermediate.zinv = 1.f / intermediate.cameraPosition.z;

            clipped.push_back(intermediate);
            clipped.push_back(pixel2);
        }

        pixel1 = pixel2;
    }

    vertexPixels = clipped;
}

void ClipEdges(vector<Pixel>& vertexPixels, vec2 edgeBounds[]) {
    int vSize = vertexPixels.size();

    vector<Pixel> clipped;
    Pixel pixel1 = vertexPixels[vSize - 1];
    Pixel pixel2;

    for (int i = 0; i < vSize; i++) {
        pixel2 = vertexPixels[i];

        if ((OnEdge(pixel1, edgeBounds)) && (OnEdge(pixel2, edgeBounds))) {
            clipped.push_back(pixel2);
        } else if (((OnEdge(pixel1, edgeBounds)) && !(OnEdge(pixel2, edgeBounds)))) {
            Pixel intermediate;
            Intermediate(pixel1, pixel2, edgeBounds, intermediate);

            clipped.push_back(intermediate);
        } else if (!(OnEdge(pixel1, edgeBounds)) && (OnEdge(pixel2, edgeBounds))) {
            Pixel intermediate;
            Intermediate(pixel1, pixel2, edgeBounds, intermediate);

            clipped.push_back(intermediate);
            clipped.push_back(pixel2);
        }

        pixel1 = pixel2;
    }

    vertexPixels = clipped;
}

bool OnEdge(Pixel pixel, vec2 edgeBounds[]) {
    /* Top edge */
    if ((edgeBounds[0].x > edgeBounds[1].x) && (pixel.y >= edgeBounds[0].y)) {
        return true;
    }
        /* Left edge */
    else if ((edgeBounds[1].y > edgeBounds[0].y) && (pixel.x >= edgeBounds[0].x)) {
        return true;
    }
        /* Right edge */
    else if ((edgeBounds[0].y > edgeBounds[1].y) && (edgeBounds[0].x >= pixel.x)) {
        return true;
    }
        /* Bottom edge */
    else if ((edgeBounds[1].x > edgeBounds[0].x) && (edgeBounds[0].y >= pixel.y)) {
        return true;
    } else {
        return false;
    }
}

void Intermediate(Pixel pixel1, Pixel pixel2, vec2 edgeBounds[], Pixel& intermediate) {
    /* Top edge */
    if (edgeBounds[0].x > edgeBounds[1].x) {
        float diff = (2 * pixel1.cameraPosition.y + pixel1.cameraPosition.z) /
                     ((2 * pixel1.cameraPosition.y + pixel1.cameraPosition.z) - (2 * pixel2.cameraPosition.y + pixel2.cameraPosition.z));
        intermediate.cameraPosition = ((1 - diff) * pixel1.cameraPosition) + (diff * pixel2.cameraPosition);
        intermediate.pos3d = ((1 - diff) * pixel1.pos3d) + (diff * pixel2.pos3d);
    }
        /* Left edge */
    else if (edgeBounds[1].y > edgeBounds[0].y) {
        float diff = (2 * pixel1.cameraPosition.x + pixel1.cameraPosition.z) /
                     ((2 * pixel1.cameraPosition.x + pixel1.cameraPosition.z) - (2 * pixel2.cameraPosition.x + pixel2.cameraPosition.z));
        intermediate.cameraPosition = ((1 - diff) * pixel1.cameraPosition) + (diff * pixel2.cameraPosition);
        intermediate.pos3d = ((1 - diff) * pixel1.pos3d) + (diff * pixel2.pos3d);
    }
        /* Right edge*/
    else if (edgeBounds[0].y > edgeBounds[1].y) {
        float diff = (2 * pixel1.cameraPosition.x - pixel1.cameraPosition.z) /
                     ((2 * pixel1.cameraPosition.x - pixel1.cameraPosition.z) - (2 * pixel2.cameraPosition.x - pixel2.cameraPosition.z));
        intermediate.cameraPosition = ((1 - diff) * pixel1.cameraPosition) + (diff * pixel2.cameraPosition);
        intermediate.pos3d = ((1 - diff) * pixel1.pos3d) + (diff * pixel2.pos3d);
    }
        /* Bottom edge*/
    else if (edgeBounds[1].x > edgeBounds[0].x) {
        float diff = (2 * pixel1.cameraPosition.y - pixel1.cameraPosition.z) /
                     ((2 * pixel1.cameraPosition.y - pixel1.cameraPosition.z) - (2 * pixel2.cameraPosition.y - pixel2.cameraPosition.z));
        intermediate.cameraPosition = ((1 - diff) * pixel1.cameraPosition) + (diff * pixel2.cameraPosition);
        intermediate.pos3d = ((1 - diff) * pixel1.pos3d) + (diff * pixel2.pos3d);
    }


    intermediate.x = (int) (camera.focalLength * (intermediate.cameraPosition.x / intermediate.cameraPosition.z)) + (SCREEN_HEIGHT / 2);
    intermediate.y = (int) (camera.focalLength * (intermediate.cameraPosition.y / intermediate.cameraPosition.z)) + (SCREEN_WIDTH / 2);
    intermediate.zinv = 1.f / intermediate.cameraPosition.z;
}

void DrawShadows(const vector<Vertex>& vertices) {
    int V = vertices.size();
    vector<Pixel> vertexPixels(V);

    for(int i=0; i<V; ++i) LightVertexShader(vertices[i], vertexPixels[i]);

    vector<Pixel> leftPixels;
    vector<Pixel> rightPixels;

    ComputePolygonRows(vertexPixels, leftPixels, rightPixels);
    ShadowRows(leftPixels, rightPixels);
}

void LightVertexShader(const Vertex& v, Pixel& pix) {
    vec4 lightPosition = (v.position - light.pos);

    pix.x = (int) (light.focalLength * (lightPosition.x / lightPosition.z)) + (SCREEN_WIDTH / 2);
    pix.y = (int) (light.focalLength * (lightPosition.y / lightPosition.z)) + (SCREEN_HEIGHT / 2);
    pix.zinv = 1.f / lightPosition.z;
}

void ShadowRows(const vector<Pixel>& leftPixels, const vector<Pixel> rightPixels) {
    int numPix = leftPixels.size();

    for (int i = 0; i < numPix; i++) {
        if (leftPixels[i].y >= 0 && leftPixels[i].y < SCREEN_WIDTH) {
            Pixel a = leftPixels[i];
            Pixel b = rightPixels[i];
            int delta_x = abs(a.x - b.x);
            int delta_y = abs(a.y - b.y);
            int pixels = glm::max(delta_x, delta_y) + 1;
            vector<Pixel> line(pixels);
            Interpolate(a, b, line);

            for (int j = 0; j < pixels; j++) {
                Pixel pix = line[j];
                int x = pix.x;
                int y = pix.y;
                float zinv = pix.zinv;
                bool pixelInBox = ((x >= 0) && (x < SCREEN_WIDTH) && (y >= 0)  && (y < SCREEN_HEIGHT));

                if (pixelInBox && (zinv >= lightBuffer[y][x])) {
                    lightBuffer[y][x] = zinv;
                }
            }
        }
    }
}