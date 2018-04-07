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
using glm::mat4x4;
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
    float yaw;
};
Camera camera;

struct Pixel {
    int x;
    int y;
    float zinv;
};

float depthBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];

/*
 * FUNCTIONS
 */

void Update();

void Draw(screen *screen, vector <Triangle> & triangles);

//void TransformationMatrix(mat4x4 M);

void InitialiseStructs();

void VertexShader(const vec4& v, Pixel& p);

void Interpolate(Pixel a, Pixel b, vector<Pixel>& result);

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
//  float dt = float(t2 - t);
    t = t2;

//    std::cout << "Render time: " << dt << " ms." << std::endl;
    /* Update variables*/

}

void InitialiseStructs() {
    /* Camera */
    camera.position = vec4(0, 0, -3.001, 1);
    camera.focalLength = SCREEN_HEIGHT;
    camera.yaw = 0;
}

void VertexShader(const vec4& v, Pixel& p) {
    vec4 v_ = v - camera.position;
    p.x = (camera.focalLength * (v_.x / v_.z)) + (SCREEN_WIDTH / 2);
    p.y = (camera.focalLength * (v_.y / v_.z)) + (SCREEN_HEIGHT / 2);
}

void Interpolate(Pixel a, Pixel b, vector<Pixel>& result) {
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
    Interpolate(a, b, line);

    for (int i = 0; i < pixels; i++) {
        PutPixelSDL(screen, line[i].x, line[i].y, color);
    }

//    ivec2 delta = glm::abs(a - b);
//    int pixels = glm::max(delta.x, delta.y) + 1;
//    vector<Pixel> line(pixels);
//    Interpolate(a, b, line);
//
//    for (int i = 0; i<pixels; i++) PutPixelSDL(screen, line[i].x, line[i].y, color);

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
        leftPixels[i].y = min_y + i;
        rightPixels[i].y = leftPixels[i].y;
    }

    for (int i = 0; i < vPSize; i++) {
        int j = (i + 1) % vPSize;
        int numPix = abs(vertexPixels[i].y - vertexPixels[j].y) + 1;
        vector<Pixel> line(numPix);
        Interpolate(vertexPixels[i], vertexPixels[j], line);

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