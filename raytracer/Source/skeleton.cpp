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

/* TODO: Extension ideas
* 1. Global Illumination
* 2. Spheres
* 3. General Models
* 4. Textures
* 5. Parallelism
* 6. anti-aliasing
* 7. Phong shading
*/

/* Extensions DONE
* 1. Soft Shadows
* 2. Specular Surfaces
* 3. Glass
* 4. Cramer's Rule
* 5. Depth of Field
* 6. Fresnel Effect ---- Needs Checking? Looks a bit strange
*/

/* ----------------------------------------------------------------------------*/
/* STRUCTS                                                                   */

struct Light{
  vec4 position;
  vec3 color;
  float width;
};
Light light;

struct Camera{
  float focalLength;
  float aperture;
  float dof_length;
  float angle;
  vec4 position;
};
Camera camera;

struct Intersection{
  vec4 position;
  float distance;
  int triangleIndex;
};

struct Options{
  int max_depth;
  int num_dof_rays;
  int num_light_rays;
};
Options options;

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */

void InitializeStructs();
void Update();
void Draw(screen* screen, const vector<Triangle>& triangles);
bool ClosestIntersection(vec4 start, vec4 dir, const vector<Triangle>& triangles, Intersection& closestIntersection );
vec3 DirectLight( const Intersection& i, const vector<Triangle>& triangles );
vec3 get_color(const Intersection& i, const vector<Triangle>& triangles, vec4 dir, int depth);
vec4 reflect(vec4 normal, vec4 dir);
vec4 refract(vec4 normal, vec4 dir, float& n, float& k);
float fresnel(vec4 normal, vec4 dir, const Intersection& i, float& n, float& k);
mat4 y_rotation(float deg);
vector<vec4> RandCameraPoints();
vector<vec4> RandLightPoints();

/* ----------------------------------------------------------------------------*/

int main( int argc, char* argv[] ){
  InitializeStructs();
  screen *screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN_MODE );

  vector<Triangle> triangles;
  LoadTestModel(triangles);

  while( NoQuitMessageSDL() ){
    Update();
    Draw(screen, triangles);
    SDL_Renderframe(screen);
    SDL_SaveImage( screen, "screenshot.png" );
  }

  KillSDL(screen);
  return 0;
}

void InitializeStructs(){
  /* Camera */
  camera.position = vec4(0.f, 0.f, -3.f, 1.0f);
  camera.focalLength = SCREEN_WIDTH;
  camera.aperture = 0.01;
  camera.dof_length = 0.008;
  camera.angle = 0;

  /* Light */
  light.position = vec4( 0, - 0.5, -0.7, 1.0 );
  light.color = 14.f * vec3( 1, 1, 1 );
  light.width = 0.2f;

  /* Options */
  options.max_depth = 5;
  options.num_dof_rays = 2;
  options.num_light_rays = 1;
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

vector<vec4> RandLightPoints(){
  vector<vec4> points;
  if(options.num_light_rays == 1){
    points.push_back(light.position);
  }
  else{
    for(int i = 0; i < options.num_light_rays; i++){
      float x = light.position.x - ((float)rand() / ((float)RAND_MAX / light.width) - (light.width/2.0f));
      float z = light.position.z + ((float)rand() / ((float)RAND_MAX / light.width) - (light.width/2.0f));
      points.push_back(vec4(x, light.position.y, z, 1.0));
    }
  }
  return points;
}

vec3 DirectLight( const Intersection& i, const vector<Triangle>& triangles){
  Triangle triangle = triangles[i.triangleIndex];
  vec3 D = vec3(0,0,0);
  vector<vec4> points = RandLightPoints();
  for(int x = 0; x < points.size(); x++){
    vec4 r = normalize(points[x] - i.position);
    float length_r = glm::length(points[x] - i.position);

    Intersection intersection;
    //in shadow
    if (ClosestIntersection(i.position + 0.001f*r, r, triangles, intersection) && (length_r >= intersection.distance)){
      //glass shadow
      if (triangles[intersection.triangleIndex].material == Glass){
        float A = 4*PI*length_r*length_r;
        vec3 B = light.color / A;

        float C = glm::dot(r, triangle.normal);
        C = glm::max(C, 0.f);

        D += 0.8f * (B * C);
      }
      else D += vec3(0,0,0);
    }
    else{
      float A = 4*PI*length_r*length_r;
      vec3 B = light.color / A;

      float C = glm::dot(r, triangle.normal);
      C = glm::max(C, 0.f);

      D += B * C;
    }
  }
  float scale = 1.f/options.num_light_rays;
  return (D*scale);
}

vec4 reflect(vec4 normal, vec4 dir){
  return dir - 2 * glm::dot(dir, normal) * normal;
}

vec4 refract (vec4 normal, vec4 dir, float& n, float& k){
  float dot = glm::dot(normal, dir);
  return n*dir + (n * dot - sqrt(k))*normal;;
}

float fresnel(vec4 normal, vec4 dir, float& n, float& k){
  float kr = 0.0f;
  float dot = glm::dot(normal, dir);
  float n1 = 1.0f;
  float n2 = 1.5f;
  if(dot > 0) std::swap(n1, n2);
  n = n1/n2;
  k = n * sqrt(std::max(0.f, 1-dot*dot));
  //total internal refrlection
  if(k >= 1) kr = 1;
  else{
    float cost = sqrt(std::max(0.f, 1 - k * k));
    dot = fabsf(dot);
    float Rs = ((n1 * dot) - (n2 * cost)) / ((n1 * dot) + (n2 * cost));
    float Rp = ((n2 * dot) - (n1 * cost)) / ((n2 * dot) + (n1 * cost));
    kr = (Rs * Rs + Rp * Rp) /2;
  }
  return kr;
}

vec3 get_color(const Intersection& i, const vector<Triangle>& triangles, vec4 dir, int depth ){
  Triangle triangle = triangles[i.triangleIndex];

  if (triangle.material == Diff){
    vec3 indirectLight = 0.5f*vec3( 1, 1, 1 );
    vec3 directLight = DirectLight(i, triangles);
    vec3 totalLight = directLight + indirectLight;
    return triangle.color * totalLight;
  }

  else if((triangle.material == Spec) && (depth < options.max_depth)){
    //reflect all light
    vec4 new_dir = reflect(triangle.normal, dir);
    Intersection closestIntersection;
    if(ClosestIntersection(i.position+(0.001f*new_dir), new_dir, triangles, closestIntersection)){
      return 0.8f * get_color(closestIntersection, triangles, new_dir, depth + 1);
    }
    else return vec3(0,0,0);
  }

  else if (triangle.material == Glass && (depth < options.max_depth)){
    // refract/reflect
    float n = 0.0f;
    float k = 0.0f;
    float kr = fresnel(triangle.normal, dir, n, k);
    vec3 refract_color;
    if(kr < 1){
      vec4 refract_dir = refract(triangle.normal, dir, n, k);
      Intersection closestIntersection;
      if(ClosestIntersection(i.position+(0.001f*refract_dir), refract_dir, triangles, closestIntersection)){
        refract_color = 0.9f * get_color(closestIntersection, triangles, refract_dir, depth+1);
      }
      else refract_color = vec3(0,0,0);
    }
    vec4 reflect_dir = reflect(triangle.normal, dir);
    vec3 reflect_color;
    Intersection intersection;
    if(ClosestIntersection(i.position+(0.001f*reflect_dir), reflect_dir, triangles, intersection)){
      reflect_color = 0.8f * get_color(intersection, triangles, reflect_dir, depth + 1);
    }
    else reflect_color = vec3(0,0,0);
    return (reflect_color * kr) + (refract_color*(1-kr));
  }
  else return vec3(0,0,0);
}

vector<vec4> RandCameraPoints(){
  vector<vec4> points;
  if(options.num_dof_rays == 1){
    points.push_back(camera.position);
  }
  else{
    for(int i = 0; i < options.num_dof_rays; i++){
      float x = camera.position.x - ((float)rand() / ((float)RAND_MAX / camera.aperture) - (camera.aperture/2.0f)) * cos(camera.angle);
      float y = camera.position.y + ((float)rand() / ((float)RAND_MAX / camera.aperture) - (camera.aperture/2.0f));
      float z = camera.position.z + ((float)rand() / ((float)RAND_MAX / camera.aperture) - (camera.aperture/2.0f)) * sin(camera.angle);
      points.push_back(vec4(x, y, z, 1.0));
    }
  }
  return points;
}

void Draw(screen* screen, const vector<Triangle>& triangles){
  /* Clear buffer */
  memset(screen->buffer, 0, screen->height*screen->width*sizeof(uint32_t));
  vector<vec4> points = RandCameraPoints();

  for(int x = 0; x<SCREEN_WIDTH; x++){
    for(int y = 0; y<SCREEN_HEIGHT; y++){
      vec3 color = vec3( 0, 0, 0 );
      Intersection closestIntersection;
      for(int i = 0; i < points.size(); i++){
        vec4 dir = y_rotation(camera.angle) * vec4(x-SCREEN_WIDTH/2, y-SCREEN_HEIGHT/2, camera.focalLength, 1);
        vec4 ray_intersection = points[i] + (dir * camera.dof_length);
        dir = ray_intersection - points[i];

        if (ClosestIntersection(points[i], dir, triangles, closestIntersection)){
          color += get_color(closestIntersection, triangles, dir, 0);
        }
      }
      float scale = 1.f/options.num_dof_rays;
      PutPixelSDL(screen, x, y, color*scale);
    }
  }
}

void Update(){
  static int t = SDL_GetTicks();
  /* Compute frame time */
  int t2 = SDL_GetTicks();
  float dt = float(t2-t);
  t = t2;
  float st = dt/1000;
  /*Good idea to remove this*/
  // std::cout << "Render time: " << st << " s." << std::endl;

  printf("Aperture = %f \n DoF Length = %f\n\n", camera.aperture, camera.dof_length);

  //camera movement
  const Uint8 *keystate = SDL_GetKeyboardState( 0 );
  if( keystate[SDL_SCANCODE_UP] ){
    camera.position.z += 0.1;
  }
  if( keystate[SDL_SCANCODE_DOWN] ){
    camera.position.z -= 0.1;
  }
  if( keystate[SDL_SCANCODE_LEFT] ){
    camera.position.x -= 0.1;
  }
  if( keystate[SDL_SCANCODE_RIGHT] ){
    camera.position.x += 0.1;
  }
  if (keystate[SDL_SCANCODE_COMMA]){
    camera.angle -= 0.02;
  }
  if( keystate[SDL_SCANCODE_PERIOD]){
    camera.angle += 0.02;
  }

  if( keystate[SDL_SCANCODE_L] ){
    camera.aperture -= 0.02;
  }
  if( keystate[SDL_SCANCODE_O] ){
    camera.aperture += 0.02;
  }
  if( keystate[SDL_SCANCODE_K] ){
    camera.dof_length -= 0.001;
  }
  if( keystate[SDL_SCANCODE_I] ){
    camera.dof_length += 0.001;
  }

  //light movement
  if( keystate[SDL_SCANCODE_R] ){
    light.position.y -= 0.1;
  }
  if( keystate[SDL_SCANCODE_F] ){
    light.position.y += 0.1;
  }
  if( keystate[SDL_SCANCODE_A] ){
    light.position.x -= 0.1;
  }
  if( keystate[SDL_SCANCODE_D] ){
    light.position.x += 0.1;
  }
  if( keystate[SDL_SCANCODE_S] ){
    light.position.z -= 0.1;
  }
  if( keystate[SDL_SCANCODE_W] ){
    light.position.z += 0.1;
  }

}

mat4 y_rotation(float angle){
  return mat4(cos(angle) , 0, sin(angle), 0,
  0          , 1, 0         , 0,
  -sin(angle), 0, cos(angle), 0,
  0          , 0, 0         , 1);
}
