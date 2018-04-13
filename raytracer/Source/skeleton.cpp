#include <iostream>
#include <math.h>
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
const int maxReflectionDepth = 5;

/* TODO: Extension ideas
* 1. Fresnel Effect
* 2. Spheres
* 3. General Models
* 4. Textures
* 5. Parallelism
* 6. anit-aliasing
*/

/* ----------------------------------------------------------------------------*/
/* STRUCTS                                                                   */

struct Light{
  vec4 position;
  vec3 color;
};
Light light;

struct Camera{
  float focalLength;
  float aperture;
  float dof_length;
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
vec3 DirectLight( const Intersection& i, const vector<Triangle>& triangles );
vec3 get_color(const Intersection& i, const vector<Triangle>& triangles, vec4 dir, int depth);
vec4 reflect(vec4 normal, vec4 dir);
vec4 refract(vec4 normal, vec4 dir);

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

  SDL_SaveImage( screen, "screenshot.png" );

  KillSDL(screen);
  return 0;
}

void InitializeStructs(){
  /* Camera */
  camera.position = vec4(0, 0, -3, 1.0);
  camera.focalLength = SCREEN_WIDTH;
  camera.aperture = 0.05;
  camera.dof_length = 0.01;
  y_rotation(0);

  /* Light */
  light.position = vec4( 0, -0.5, -0.7, 1.0 );
  light.color = 14.f * vec3( 1, 1, 1 );
}

bool ClosestIntersection(vec4 start, vec4 dir, const vector<Triangle>& triangles, Intersection& closestIntersection){
  bool intersect = false;
  closestIntersection.distance = std::numeric_limits<float>::max();

  for(int i=0; i<triangles.size(); i++){
    vec3 e1 = vec3(triangles[i].v1.x - triangles[i].v0.x, triangles[i].v1.y - triangles[i].v0.y, triangles[i].v1.z - triangles[i].v0.z);
    vec3 e2 = vec3(triangles[i].v2.x - triangles[i].v0.x, triangles[i].v2.y - triangles[i].v0.y, triangles[i].v2.z - triangles[i].v0.z);
    vec3 b = vec3(start.x - triangles[i].v0.x, start.y - triangles[i].v0.y, start.z - triangles[i].v0.z);

    vec3 d = vec3(dir.x, dir.y, dir.z);
    mat3 A(-d, e1, e2);

    float detA = glm::determinant(A);
    float det = glm::determinant(mat3(b, e1, e2));
    float t = det / detA;

    if(t>=0 && t < closestIntersection.distance){
      det = glm::determinant(mat3(-d, b, e2));
      float u = det / detA;
      if(u > 0 ){
        det = glm::determinant(mat3(-d, e1, b));
        float v = det/detA;
        if(v >= 0 && (u + v) <= 1){
          intersect = true;
          closestIntersection.triangleIndex = i;
          closestIntersection.distance = t;
          closestIntersection.position = triangles[i].v0 + (u * vec4(e1.x,e1.y,e1.z, 0)) + (v * vec4(e2.x,e2.y,e2.z, 0));
        }
      }
    }
  }
  return intersect;
}

vec3 DirectLight( const Intersection& i, const vector<Triangle>& triangles){
  Triangle triangle = triangles[i.triangleIndex];
  vec3 D = vec3(0,0,0);

  for(float x = -0.1; x < 0.1; x+=0.04){
    for(float z = -0.1; z < 0.1; z+=0.04){
      vec4 light_position = vec4(light.position.x+x, light.position.y, light.position.z+z, 1.0);
      vec4 r = normalize(light_position - i.position);
      float length_r = glm::length(light_position - i.position);

      Intersection intersection;
      //TODO: update this to absorb a small amount of the light
      if (ClosestIntersection(i.position + 0.001f*r, r, triangles, intersection) && (length_r >= intersection.distance)){
        if (triangles[intersection.triangleIndex].material == Glass){
          float A = 4*PI*length_r*length_r;
          vec3 B = light.color / A;

          float C = dot(r, triangle.normal);
          C = glm::max(C, 0.f);

          D += 0.8f * (B * C);
        }
        else D += vec3(0,0,0);
      }
      else{
        float A = 4*PI*length_r*length_r;
        vec3 B = light.color / A;

        float C = dot(r, triangle.normal);
        C = glm::max(C, 0.f);

        D += B * C;
      }
    }
  }
  return (D*0.04f);
}

vec4 reflect(vec4 normal, vec4 dir){
  return dir - 2 * dot(dir, normal) * normal;
}

vec4 refract(vec4 normal, vec4 dir){
  float dot = glm::dot(normal, dir);
  float n1 = 1.0f;
  float n2 = 1.5f;
  if(dot < 0){
    dot = - dot;
  }
  else{
    std::swap(n1, n2);
    normal = -normal;
  }
  float n = n1 / n2;
  float k = (1-(n*n) * (1-(dot*dot)));
  if(k < 0){
    printf("here\n");
    return reflect(normal, dir);
  }
  else{
    return n*dir + (n * dot - sqrt(k))*normal;;
  }
}

vec3 get_color(const Intersection& i, const vector<Triangle>& triangles, vec4 dir, int depth ){
  Triangle triangle = triangles[i.triangleIndex];
  if (triangle.material == Diff){
    vec3 indirectLight = 0.5f*vec3( 1, 1, 1 );
    vec3 directLight = DirectLight(i, triangles);
    vec3 totalLight = directLight + indirectLight;
    return triangle.color * totalLight;
  }

  else if((triangle.material == Spec) && (depth < maxReflectionDepth)){
    //reflect all light
    vec4 new_dir = reflect(triangles[i.triangleIndex].normal, dir);
    Intersection closestIntersection;
    if(ClosestIntersection(i.position+(0.001f*new_dir), new_dir, triangles, closestIntersection)){
      return 0.8f * get_color(closestIntersection, triangles, new_dir, depth + 1);
    }
    else return vec3(0,0,0);
  }

  else if (triangle.material == Glass && (depth < maxReflectionDepth)){
    // refract/reflect
    vec4 new_dir = refract(triangles[i.triangleIndex].normal, dir);
    Intersection closestIntersection;
    if(ClosestIntersection(i.position+(0.001f*new_dir), new_dir, triangles, closestIntersection)){
      return 0.9f * get_color(closestIntersection, triangles, new_dir, depth + 1);
    }
    else return vec3(0,0,0);
  }
  else return vec3(0,0,0);
}

void Draw(screen* screen, const vector<Triangle>& triangles){
  /* Clear buffer */
  memset(screen->buffer, 0, screen->height*screen->width*sizeof(uint32_t));

  for(int x = 0; x<SCREEN_WIDTH; x++){
    for(int y = 0; y<SCREEN_HEIGHT; y++){
      // printf("(%d, %d)", x, y);
      vec3 color = vec3( 0, 0, 0 );
      Intersection closestIntersection;

      for (float i = -camera.aperture; i < camera.aperture; i+=0.01){
        for(float j = -camera.aperture; j < camera.aperture; j+=0.01){
          vec4 start = camera.position;
          vec4 dir = camera.rotation * vec4(x-SCREEN_WIDTH/2, y-SCREEN_HEIGHT/2, camera.focalLength, 1);
          vec4 ray_intersection = start + (dir * camera.dof_length);
          start.x += i;
          start.y += j;
          dir = ray_intersection - start;

          if (ClosestIntersection(start, dir, triangles, closestIntersection)){
            color += get_color(closestIntersection, triangles, dir, 0);
          }
        }
      }
      PutPixelSDL(screen, x, y, color*0.01f);
    }
  }
}

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
