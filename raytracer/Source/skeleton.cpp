#include <iostream>
#include<math.h>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModelH.h"
#include <stdint.h>
#include <limits.h>

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

#define SCREEN_WIDTH 300
#define SCREEN_HEIGHT 300
#define FULLSCREEN_MODE false
#define PI 3.14159265

/* ----------------------------------------------------------------------------*/
/* STRUCTS                                                                   */

struct Light{
  vec4 position;
  vec3 color;
};
Light light;

struct Camera{
  float focalLength;
  vec4 position;
  mat4 rotation;
};
Camera camera;

struct Intersection{
  vec4 position;
  float distance;
  int triangleIndex;
};

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */

void InitializeStructs();
void Update();
void Draw(screen* screen, const vector<Triangle>& triangles);
bool ClosestIntersection(vec4 start, vec4 dir, const vector<Triangle>& triangles, Intersection& closestIntersection );
void y_rotation(float deg);
vec3 DirectLight( const Intersection& i, const vector<Triangle>& triangles, bool& shadow);

/* ----------------------------------------------------------------------------*/

int main( int argc, char* argv[] ){
  screen *screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN_MODE );
  InitializeStructs();

  vector<Triangle> triangles;
  LoadTestModel(triangles);

  while( NoQuitMessageSDL() ){
    Update();
    Draw(screen, triangles);
    SDL_Renderframe(screen);
  }

  SDL_SaveImage( screen, "screenshot.bmp" );

  KillSDL(screen);
  return 0;
}

void InitializeStructs(){
  /* Camera */
  camera.position = vec4(0, 0, -3, 1.0);
  camera.focalLength = SCREEN_WIDTH;
  y_rotation(0);

  /* Light */
  light.position = vec4( 0, -0.5, -0.7, 1.0 );
  light.color = 14.f * vec3( 1, 1, 1 );
}

bool ClosestIntersection(vec4 start, vec4 dir, const vector<Triangle>& triangles, Intersection& closestIntersection){
  bool intersect = false;
  closestIntersection.distance = std::numeric_limits<float>::max();

  for(int i=0; i<triangles.size(); i++){
    vec4 v0 = triangles[i].v0;
    vec4 v1 = triangles[i].v1;
    vec4 v2 = triangles[i].v2;

    vec3 e1 = vec3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    vec3 e2 = vec3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    vec3 b = vec3(start.x - v0.x, start.y - v0.y, start.z - v0.z);

    vec3 d = vec3(dir.x, dir.y, dir.z);

    mat3 A(-d, e1, e2);
    vec3 x = glm::inverse( A ) * b;

    if(x.x >= 0 && x.y >= 0 && x.z >= 0 && (x.y + x.z) <= 1 && x.x < closestIntersection.distance){
      intersect = true;
      closestIntersection.triangleIndex = i;
      closestIntersection.distance = x.x;
      closestIntersection.position = v0 + (x.y * vec4(e1.x,e1.y,e1.z, 0)) + (x.z * vec4(e2.x,e2.y,e2.z, 0));
    }
  }
  return intersect;
}

vec3 DirectLight( const Intersection& i, const vector<Triangle>& triangles, bool& inShadow ){
  Triangle triangle = triangles[i.triangleIndex];
  vec4 r = normalize(light.position - i.position);
  float length_r = glm::length(light.position - i.position);

  Intersection intersection;
  if (ClosestIntersection(i.position + 0.001f*r, r, triangles, intersection)){
    if(length_r >= intersection.distance) inShadow = true;
    else inShadow = false;
  }else inShadow = false;

  float A = 4*PI*length_r*length_r;
  vec3 B = light.color / A;

  float C = dot(r, triangle.normal);
  C = glm::max(C, 0.f);

  vec3 D = B * C;

  return (D);
}

void Draw(screen* screen, const vector<Triangle>& triangles){
  /* Clear buffer */
  memset(screen->buffer, 0, screen->height*screen->width*sizeof(uint32_t));

  for(int x = 0; x<SCREEN_WIDTH; x++){
    for(int y = 0; y<SCREEN_HEIGHT; y++){
      vec4 start = camera.position;
      vec4 dir = vec4(x-SCREEN_WIDTH/2, y-SCREEN_HEIGHT/2, camera.focalLength, 1) ;
      Intersection closestIntersection;

      vec3 indirectLight = 0.5f*vec3( 1, 1, 1 );
      vec3 directLight = vec3( 0, 0, 0 );
      vec3 color = vec3( 0, 0, 0 );

      if (ClosestIntersection(start, camera.rotation * dir, triangles, closestIntersection)){
        bool inShadow;
        directLight = DirectLight(closestIntersection, triangles, inShadow);
        if (inShadow) directLight = vec3(0,0,0);
        vec3 totalLight = directLight + indirectLight;
        color = triangles[closestIntersection.triangleIndex].color * totalLight;
        PutPixelSDL(screen, x, y, color);
      }
    }
  }
}

/* TODO Rotation not quite working correctly */
float yaw = 0.0;
void Update(){
  static int t = SDL_GetTicks();
  /* Compute frame time */
  int t2 = SDL_GetTicks();
  float dt = float(t2-t);
  t = t2;
  /*Good idea to remove this*/
  std::cout << "Render time: " << dt << " ms." << std::endl;

  //camera movement
  const Uint8 *keystate = SDL_GetKeyboardState( 0 );
  if( keystate[SDL_SCANCODE_UP] ){
    camera.position.y-=0.1;
  }
  if( keystate[SDL_SCANCODE_DOWN] ){
    camera.position.y+=0.1;
  }
  if( keystate[SDL_SCANCODE_LEFT] ){
    yaw -= 0.02;
    y_rotation(yaw);
    // camera.position = camera.rotation * camera.position;
  }
  if( keystate[SDL_SCANCODE_RIGHT] ){
    yaw += 0.02;
    y_rotation(yaw);
    // camera.position = camera.rotation * camera.position;
  }

  //light movement
  if( keystate[SDL_SCANCODE_W] ){
    light.position.y -= 0.1;
  }
  if( keystate[SDL_SCANCODE_S] ){
    light.position.y += 0.1;
  }
  if( keystate[SDL_SCANCODE_A] ){
    light.position.x -= 0.1;
  }
  if( keystate[SDL_SCANCODE_D] ){
    light.position.x += 0.1;
  }
}

void y_rotation(float angle){
  camera.rotation = mat4(cos(angle) , 0, sin(angle), 0,
                         0          , 1, 0         , 0,
                         -sin(angle), 0, cos(angle), 0,
                         0          , 0, 0         , 1);
}
