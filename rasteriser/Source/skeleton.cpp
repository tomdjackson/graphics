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
};

float depthBuffer[SCREEN_WIDTH][SCREEN_HEIGHT];

mat4 identityMatrix = mat4(1.0f);

const float rotationIncr = 0.1;
const float translationIncr = 0.5;

/*
 * FUNCTIONS
 */

void Update();

void Draw(screen *screen, vector <Triangle> & triangles);

void Rotate();

void InitialiseStructs();

void InitialiseDepthBuffer();

void VertexShader(const vec4& v, Pixel& p);

void InterpolatePixel(Pixel a, Pixel b, vector<Pixel>& result);

void DrawLineSDL(Pixel a, Pixel b, vec3 color, screen *screen);

void ComputePolygonRows(const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels);

void DrawPolygonRows(const vector<Pixel>& leftPixels, const vector<Pixel> rightPixels, vec3 color, screen* screen);

void DrawPolygon(const vector<vec4>& vertices, vec3 color, screen* screen);

int main(int argc, char *argv[]) {
    screen *screen = InitializeSDL(SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN_MODE);

    InitialiseStructs();

    vector<Triangle> triangles;
    LoadTestModel(triangles);

    while (NoQuitMessageSDL()) {
        Update();
        Draw(screen, triangles);
        SDL_Renderframe(screen);
    }

    SDL_SaveImage(screen, "screenshot.bmp");

    KillSDL(screen);
    return 0;
}

void Draw(screen* screen, vector<Triangle>&triangles) {
    /* Clear buffer */
    memset(screen->buffer, 0, screen->height * screen->width * sizeof(uint32_t));
    InitialiseDepthBuffer();

    for (uint32_t i = 0; i<triangles.size(); ++i) {
        vector<vec4> vertices(3);

        vertices[0] = triangles[i].v0;
        vertices[1] = triangles[i].v1;
        vertices[2] = triangles[i].v2;
        vec3 color = triangles[i].color;

        DrawPolygon(vertices, color, screen);
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
    if (keyPress[SDL_SCANCODE_UP]) {
        camera.position += translationIncr * up;
//        Rotate();
    }
    if (keyPress[SDL_SCANCODE_DOWN]) {
        camera.position -= translationIncr * up;
//        Rotate();
    }
    if (keyPress[SDL_SCANCODE_O]) {
        camera.position += translationIncr * forward;
//        Rotate();
    }
    if (keyPress[SDL_SCANCODE_K]) {
        camera.position -= translationIncr * forward;
//        Rotate();
    }
    if (keyPress[SDL_SCANCODE_RIGHT]) {
        camera.position += translationIncr * right;
//        Rotate();
    }
    if (keyPress[SDL_SCANCODE_LEFT]) {
        camera.position -= translationIncr * right;
//        Rotate();
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
    if (keyPress[SDL_SCANCODE_R]) {
        camera.position = vec4(0, 0, -3.001, 1);
        camera.yaw = 0;
        camera.R = identityMatrix;
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
}

void InitialiseDepthBuffer() {
    for(int i=0; i < SCREEN_HEIGHT; i++)
        for(int j=0; j < SCREEN_WIDTH; j++)
            depthBuffer[j][i] = 0;
}

void VertexShader(const vec4& v, Pixel& pix) {
    vec4 point = (v - camera.position) * camera.R;

    pix.zinv = 1/point.z;
    pix.x = (int) (camera.focalLength * (point.x / point.z)) + (SCREEN_WIDTH / 2);
    pix.y = (int) (camera.focalLength * (point.y / point.z)) + (SCREEN_HEIGHT / 2);
}

void InterpolatePixel(Pixel a, Pixel b, vector<Pixel>& result) {
    int N = result.size();

    vec3 step((b.x - a.x),(b.y - a.y), (b.zinv - a.zinv));
    step = step / float(max(N-1,1));
    vec3 current(a.x, a.y, a.zinv);

    for (int i = 0; i < N; i++) {
        result[i].x = current.x;
        result[i].y = current.y;
        result[i].zinv = current.z;
        current += step;
    }
}

void DrawLineSDL(Pixel a, Pixel b, vec3 color, screen *screen) {
    int delta_x = abs(a.x - b.x);
    int delta_y = abs(a.y - b.y);
    int pixels = glm::max(delta_x, delta_y) + 1;
    vector<Pixel> line(pixels);
    InterpolatePixel(a, b, line);

    for (int i = 0; i < pixels; i++) {
        int x = line[i].x;
        int y = line[i].y;
        if (line[i].zinv > depthBuffer[x][y]) {
            depthBuffer[x][y] = line[i].zinv;
            PutPixelSDL(screen, line[i].x, line[i].y, color);
        }

    }
}

void ComputePolygonRows(const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels) {
    int min_y = +numeric_limits<int>::max();
    int max_y = -numeric_limits<int>::max();

    int vPSize = vertexPixels.size();

    for (int i = 0; i < vPSize; i++) {
        if (vertexPixels[i].y > max_y) max_y = vertexPixels[i].y;
        if (vertexPixels[i].y < min_y) min_y = vertexPixels[i].y;
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
        InterpolatePixel(vertexPixels[i], vertexPixels[j], line);

        for (int k = 0; k < numPix; k++) {
            int x = line[k].x;
            int y_ = line[k].y - min_y;

            if (x > rightPixels[y_].x) rightPixels[y_] = line[k];
            if (x < leftPixels[y_].x) leftPixels[y_] = line[k];
        }
    }
}

void DrawPolygonRows(const vector<Pixel>& leftPixels, const vector<Pixel> rightPixels, vec3 color, screen* screen) {
    int numPix = leftPixels.size();

    for (int i = 0; i < numPix; i++) DrawLineSDL(leftPixels[i], rightPixels[i], color, screen);
}

void DrawPolygon(const vector<vec4>& vertices, vec3 color, screen* screen) {
    int V = vertices.size();
    vector<Pixel> vertexPixels(V);

    for( int i=0; i<V; ++i ) VertexShader(vertices[i], vertexPixels[i]);

    vector<Pixel> leftPixels;
    vector<Pixel> rightPixels;

    ComputePolygonRows(vertexPixels, leftPixels, rightPixels);
    DrawPolygonRows(leftPixels, rightPixels, color, screen);
}