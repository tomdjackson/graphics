#include <iostream>
#include <math.h>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModelH.h"
#include <stdint.h>
#include <limits.h>
#include "glm/ext.hpp"

using namespace std;
using std::vector;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;
using glm::vec2;

#define SCREEN_WIDTH 400
#define SCREEN_HEIGHT 400
#define FULLSCREEN_MODE false
#define PI 3.14159265

/* TODO: Extension ideas
* 1. Global Illumination
* 2. Phong shading
* 3. General Models
* 4. Textures
* 6. anti-aliasing
* 7. water
*/

/* Extensions DONE
* 1. Soft Shadows
* 2. Specular Surfaces
* 3. Glass
* 4. Cramer's Rule
* 5. Depth of Field
* 6. Fresnel Effect
* 7. Spheres
* 8. Parallelism
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
  int objectIndex;
};

struct Options{
  int max_depth;
  int num_dof_rays;
  int num_light_rays;
};
Options options;

vector<Object*> objects;

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */

void InitializeStructs();
void Update();
void Draw(screen* screen);
bool ClosestIntersection(vec4 start, vec4 dir, Intersection& closestIntersection );
vec3 DirectLight( const Intersection& i);
vec3 get_color(const Intersection& i, vec4 dir, int depth);
vec4 reflect(vec4 normal, vec4 dir);
vec4 refract(vec4 normal, vec4 dir, float& n, float& k);
float fresnel(vec4 normal, vec4 dir, const Intersection& i, float& n, float& k);
mat4 y_rotation(float deg);
void RandCameraPoints(vector<vec4> &points);
void RandLightPoints(vector<vec4> &points);
vec3 fxaa(int x, int y);
vec3 getFromBuffer(float x, float y);
vec3 getFromBuffer(vec2 point);

/* ----------------------------------------------------------------------------*/

int main( int argc, char* argv[] ){
  InitializeStructs();
  screen *screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN_MODE );

  LoadTestModel(objects);

  while( NoQuitMessageSDL() ){
    Update();
    Draw(screen);
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
  light.color = 8.f * vec3( 1, 1, 1 );
  light.width = 0.2f;

  /* Options */
  options.max_depth = 10;
  options.num_dof_rays = 1;
  options.num_light_rays = 1;
}

bool ClosestIntersection(vec4 start, vec4 dir, Intersection& closestIntersection){
  bool intersect = false;
  closestIntersection.distance = std::numeric_limits<float>::max();
  float t = closestIntersection.distance;
  vec4 position = vec4(0.f,0.f,0.f,1.f);
  for(int i = 0; i < objects.size(); i++){
    if(objects[i]->intersect(start, dir, t, position)){
      if (t < closestIntersection.distance){
        intersect = true;
        closestIntersection.objectIndex = i;
        closestIntersection.distance = t;
        closestIntersection.position = position;
      }
    }
  }
  return intersect;
}

void RandLightPoints(vector<vec4> &points){
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
}

vec3 DirectLight( const Intersection& i){
  vec3 D = vec3(0,0,0);
  vector<vec4> points;
  RandLightPoints(points);
  for(int x = 0; x < points.size(); x++){
    glm::vec4 r = points[x] - i.position;
    glm::vec3 r3 = glm::vec3(r.x, r.y, r.z);
    r3 = glm::normalize(r3);
    r = glm::vec4(r3.x, r3.y, r3.z, 1.f);
    float length_r = glm::length(points[x] - i.position);

    Intersection intersection;
    //in shadow
    if (ClosestIntersection(i.position + 0.0001f*r, r, intersection) && (length_r > intersection.distance)){
      //glass shadow
      if (objects[intersection.objectIndex]->material == Glass){
        float A = 4*PI*length_r*length_r;
        vec3 B = light.color / A;
        vec4 normal = objects[i.objectIndex]->computeNormal( i.position );
        float C = glm::dot(r, normal);
        C = glm::max(C, 0.f);

        D += (B * C) * 0.8f;
      }
      else D += vec3(0,0,0);
    }
    else{
      float A = 4*PI*length_r*length_r;
      vec3 B = light.color / A;
      vec4 normal = objects[i.objectIndex]->computeNormal( i.position );
      float C = glm::dot(r, normal);
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
  float cosi = glm::clamp(-1.f,1.f,glm::dot(dir, normal));
  float n1 = 1.5f, n2 = 1.f;
  if (cosi < 0) cosi = -cosi;
  else {
    std::swap(n1, n2);
    normal = -normal;
  }
  n = n1 / n2;
  k = 1 - n * n * (1 - cosi * cosi);
  return (k < 0) ? vec4() : n * dir + (n * cosi - sqrtf(k)) * normal;
}

float fresnel(vec4 normal, vec4 dir, float& n, float& k){
  float kr = 0.0f;
  float dot = glm::clamp(1.f,1.f,glm::dot(normal, dir));
  float n1 = 1.5f, n2 = 1.f;
  if(dot > 0) std::swap(n1, n2);
  n = n1/n2;
  k = n * sqrt(std::max(0.f, 1-dot*dot));
  //total internal refrlection
  if(k >= 1) kr = 1;
  else{
    float cost = sqrtf(std::max(0.f, 1 - k * k));
    dot = fabsf(dot);
    float Rs = ((n1 * dot) - (n2 * cost)) / ((n1 * dot) + (n2 * cost));
    float Rp = ((n2 * dot) - (n1 * cost)) / ((n2 * dot) + (n1 * cost));
    kr = (Rs * Rs + Rp * Rp) /2;
  }
  return kr;
}

vec3 get_color(const Intersection& i, vec4 dir, int depth ){
  if (objects[i.objectIndex]->material == Diff){
    vec3 indirectLight = 0.5f*vec3( 1, 1, 1 );
    vec3 directLight = DirectLight(i);
    return objects[i.objectIndex]->color * (directLight + indirectLight);
  }

  else if((objects[i.objectIndex]->material == Spec) && (depth < options.max_depth)){
    //reflect all light
    vec4 new_dir = reflect(objects[i.objectIndex]->computeNormal( i.position ), dir);
    Intersection closestIntersection;
    if(ClosestIntersection(i.position+(0.00001f*new_dir), new_dir, closestIntersection)){
      depth++;
      return 0.8f * get_color(closestIntersection, new_dir, depth);
    }
    else return vec3(0,0,0);
  }

  else if (objects[i.objectIndex]->material == Glass && (depth < options.max_depth)){
    // refract/reflect
    float n = 0.0f;
    float k = 0.0f;
    vec4 normal = objects[i.objectIndex]->computeNormal( i.position );
    float kr = fresnel(normal, dir, n, k);
    vec3 refract_color;
    if(kr < 1){
      vec4 refract_dir = refract(normal, dir, n, k);
      Intersection closestIntersection;
      if(ClosestIntersection(i.position+(0.00001f*refract_dir), refract_dir, closestIntersection)){
        refract_color = 0.8f * get_color(closestIntersection, refract_dir, depth+1);
      }
      else refract_color = vec3(0,0,0);
    }
    vec4 reflect_dir = reflect(normal, dir);
    vec3 reflect_color = vec3(0,0,0);
    Intersection intersection;
    if(ClosestIntersection(i.position+(0.00001f*reflect_dir), reflect_dir, intersection)){
      reflect_color = 0.8f * get_color(intersection, reflect_dir, depth + 1);
    }
    return (reflect_color * kr) + (refract_color*(1-kr));
  }
  else return vec3(0,0,0);
}

void RandCameraPoints(vector<vec4> &points){
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
}

void Draw(screen* screen){
  /* Clear buffer */
  memset(screen->buffer, 0, screen->height*screen->width*sizeof(uint32_t));
  vector<vec4> points;
  RandCameraPoints(points);

  #pragma omp parallel for
  for(int x = 0; x<SCREEN_WIDTH; x++){
    for(int y = 0; y<SCREEN_HEIGHT; y++){
      vec3 color = vec3( 0, 0, 0 );
      Intersection closestIntersection;
      for(int i = 0; i < points.size(); i++){
        vec4 dir = y_rotation(camera.angle) * vec4(x-SCREEN_WIDTH/2, y-SCREEN_HEIGHT/2, camera.focalLength, 1);
        vec4 ray_intersection = camera.position + (dir * camera.dof_length);
        ray_intersection.w = 1.f;
        dir = ray_intersection - points[i];

        if (ClosestIntersection(points[i], dir, closestIntersection)){
          color += get_color(closestIntersection, dir, 0);
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
  std::cout << "Render time: " << st << " s." << std::endl;

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
    camera.angle += 0.02;
  }
  if( keystate[SDL_SCANCODE_PERIOD]){
    camera.angle -= 0.02;
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
