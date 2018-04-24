#ifndef TEST_MODEL_CORNEL_BOX_H
#define TEST_MODEL_CORNEL_BOX_H

// Defines a simple test model: The Cornel Box

#include <glm/glm.hpp>
#include <vector>
#include "glm/ext.hpp"

enum Material { Spec, Diff, Glass};

class Object {
public:
  glm::vec3 color;
  Material material;

  Object(glm::vec3 color, Material material): color(color), material(material){} // constructor
  virtual ~Object() {} // virtual destructor
  virtual bool intersect(glm::vec4 orig, glm::vec4 dir, float &t, glm::vec4 &position) const = 0;
  virtual void scale(float L) = 0;
  virtual glm::vec4 computeNormal(glm::vec4 position) const = 0;
};

bool solveQuadratic(const float &a, const float &b, const float &c, float &x0, float &x1)
{
    float discr = b * b - 4 * a * c;
    if (discr < 0) return false;
    else if (discr == 0) {
        x0 = x1 = - 0.5 * b / a;
    }
    else {
        float q = (b > 0) ?
            -0.5 * (b + sqrt(discr)) :
            -0.5 * (b - sqrt(discr));
        x0 = q / a;
        x1 = c / q;
    }

    return true;
}

class Sphere : public Object{
public:
  float radius;
  float radius2;
  glm::vec4 center;

  Sphere(glm::vec4 c, float r, glm::vec3 color, Material material): Object(color, material), radius(r), radius2(r*r), center(c){}

  bool intersect(glm::vec4 orig, glm::vec4 dir, float &t, glm::vec4 &position) const{
    float t0, t1;
    glm::vec4 L = orig - center;
    float a = glm::dot(dir, dir);
    float b = 2 * glm::dot(dir, L);
    float c = glm::dot(L, L) - radius2;
    if (!solveQuadratic(a, b, c, t0, t1)) return false;

    if (t0 > t1) std::swap(t0, t1);

    if (t0 < 0) {
      t0 = t1;
      if (t0 < 0) return false;
    }
    t = t0;
    position = orig + (dir * t);
    return true;
  }

  glm::vec4 computeNormal( glm::vec4 position) const{
    glm::vec4 normal = position - center;
    glm::vec3 normal3 = glm::vec3(normal.x, normal.y, normal.z);
    normal3 = glm::normalize(normal3);
    normal = glm::vec4(normal3.x, normal3.y, normal3.z, 1.f);
    return normal;
  }

  void scale(float L){
    // center *= L/2;
    // center -= glm::vec4(1,1,1,1);
    // center.x *= -1;
    // center.y *= -1;
    // center.w = 1.0;
    //
    // radius *= L/2;
    // radius -= 1;
    // radius *= -1;
    // radius2 = radius * radius;
  }
};

// Used to describe a triangular surface:
class Triangle : public Object{
public:
  glm::vec4 v0;
  glm::vec4 v1;
  glm::vec4 v2;

  Triangle( glm::vec4 v0, glm::vec4 v1, glm::vec4 v2, glm::vec3 color, Material material):
  Object(color, material), v0(v0), v1(v1), v2(v2) {}

  bool intersect(glm::vec4 start, glm::vec4 dir, float &t, glm::vec4 &position) const{
    bool intersect = false;
    glm::vec3 e1 = glm::vec3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    glm::vec3 e2 = glm::vec3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    glm::vec3 b = glm::vec3(start.x - v0.x, start.y - v0.y, start.z - v0.z);

    glm::vec3 d = glm::vec3(dir.x, dir.y, dir.z);
    glm::mat3 A(-d, e1, e2);

    float detA = glm::determinant(A);
    float det = glm::determinant(glm::mat3(b, e1, e2));
    t = det / detA;

    if(t>=0){
      det = glm::determinant(glm::mat3(-d, b, e2));
      float u = det / detA;
      if(u >= 0 ){
        det = glm::determinant(glm::mat3(-d, e1, b));
        float v = det/detA;
        if(v >= 0 && (u + v) <= 1){
          intersect = true;
          position = v0 + (u * glm::vec4(e1.x,e1.y,e1.z, 0)) + (v * glm::vec4(e2.x,e2.y,e2.z, 0));
        }
      }
    }
    return intersect;
  }

  glm::vec4 computeNormal(glm::vec4 intersect) const  {
    glm::vec3 e1 = glm::vec3(v1.x-v0.x,v1.y-v0.y,v1.z-v0.z);
    glm::vec3 e2 = glm::vec3(v2.x-v0.x,v2.y-v0.y,v2.z-v0.z);
    glm::vec3 normal3 = glm::normalize( glm::cross( e2, e1 ) );
    glm::vec4 normal;
    normal.x = normal3.x;
    normal.y = normal3.y;
    normal.z = normal3.z;
    normal.w = 1.0;
    return normal;
  }

  void scale(float L) {
    v0 *= 2/L;
    v1 *= 2/L;
    v2 *= 2/L;

    v0 -= glm::vec4(1,1,1,1);
    v1 -= glm::vec4(1,1,1,1);
    v2 -= glm::vec4(1,1,1,1);

    v0.x *= -1;
    v1.x *= -1;
    v2.x *= -1;

    v0.y *= -1;
    v1.y *= -1;
    v2.y *= -1;

    v0.w = 1.0;
    v1.w = 1.0;
    v2.w = 1.0;
  }
};

// Loads the Cornell Box. It is scaled to fill the volume:
// -1 <= x <= +1
// -1 <= y <= +1
// -1 <= z <= +1
void LoadTestModel( std::vector<Object*>& objects )
{
  using glm::vec3;
  using glm::vec4;

  // Defines colors:
  vec3 red(    0.75f, 0.15f, 0.15f );
  vec3 yellow( 0.75f, 0.75f, 0.15f );
  vec3 green(  0.15f, 0.75f, 0.15f );
  vec3 cyan(   0.15f, 0.75f, 0.75f );
  vec3 blue(   0.15f, 0.15f, 0.75f );
  vec3 purple( 0.75f, 0.15f, 0.75f );
  vec3 white(  0.75f, 0.75f, 0.75f );

  objects.clear();
  objects.reserve( 5*2*3 );

  // ---------------------------------------------------------------------------
  // Room

  float L = 555;			// Length of Cornell Box side.

  vec4 A(L,0,0,1);
  vec4 B(0,0,0,1);
  vec4 C(L,0,L,1);
  vec4 D(0,0,L,1);

  vec4 E(L,L,0,1);
  vec4 F(0,L,0,1);
  vec4 G(L,L,L,1);
  vec4 H(0,L,L,1);

  //Spheres
  objects.push_back( new Sphere(glm::vec4(-1,0,0,1), 0.35, red, Spec));
  objects.push_back( new Sphere(glm::vec4(0,0,0,1), 0.25, red, Glass));

  // Floor:
  objects.push_back( new Triangle( C, B, A, green, Diff ));
  objects.push_back( new Triangle( C, D, B, green, Diff) );

  // Left wall
  objects.push_back( new Triangle( A, E, C, purple, Diff));
  objects.push_back( new Triangle( C, E, G, purple, Diff));

  // Right wall
  objects.push_back( new Triangle( F, B, D, yellow, Spec));
  objects.push_back( new Triangle( H, F, D, yellow, Spec));

  // Ceiling
  objects.push_back( new Triangle( E, F, G, cyan, Diff) );
  objects.push_back( new Triangle( F, H, G, cyan, Diff) );

  // Back wall
  objects.push_back( new Triangle( G, D, C, white, Diff) );
  objects.push_back( new Triangle( G, H, D, white, Diff) );

  // ---------------------------------------------------------------------------
  // Short block

  A = vec4(290,0,114,1);
  B = vec4(130,0, 65,1);
  C = vec4(240,0,272,1);
  D = vec4( 82,0,225,1);

  E = vec4(290,165,114,1);
  F = vec4(130,165, 65,1);
  G = vec4(240,165,272,1);
  H = vec4( 82,165,225,1);

  // Front
  objects.push_back( new Triangle(E,B,A,red, Diff)) ;
  objects.push_back( new Triangle(E,F,B,red, Diff)) ;

  // Front
  objects.push_back( new Triangle(F,D,B,red, Diff)) ;
  objects.push_back( new Triangle(F,H,D,red, Diff)) ;

  // BACK
  objects.push_back( new Triangle(H,C,D,red, Diff)) ;
  objects.push_back( new Triangle(H,G,C,red, Diff)) ;

  // LEFT
  objects.push_back( new Triangle(G,E,C,red, Diff)) ;
  objects.push_back( new Triangle(E,A,C,red, Diff)) ;

  // TOP
  objects.push_back( new Triangle(G,F,E,red, Diff)) ;
  objects.push_back( new Triangle(G,H,F,red, Diff)) ;

  // ---------------------------------------------------------------------------
  // Tall block

  A = vec4(423,0,247,1);
  B = vec4(265,0,296,1);
  C = vec4(472,0,406,1);
  D = vec4(314,0,456,1);

  E = vec4(423,330,247,1);
  F = vec4(265,330,296,1);
  G = vec4(472,330,406,1);
  H = vec4(314,330,456,1);

  // Front
  objects.push_back( new Triangle(E,B,A,blue, Glass) );
  objects.push_back( new Triangle(E,F,B,blue, Glass) );

  // Front
  objects.push_back( new Triangle(F,D,B,blue, Glass) );
  objects.push_back( new Triangle(F,H,D,blue, Glass) );

  // BACK
  objects.push_back( new Triangle(H,C,D,blue, Glass) );
  objects.push_back( new Triangle(H,G,C,blue, Glass) );

  // LEFT
  objects.push_back( new Triangle(G,E,C,blue, Glass) );
  objects.push_back( new Triangle(E,A,C,blue, Glass) );

  // TOP
  objects.push_back( new Triangle(G,F,E,blue, Glass) );
  objects.push_back( new Triangle(G,H,F,blue, Glass) );


  // ----------------------------------------------
  // Scale to the volume [-1,1]^3

  for( int i=0; i<objects.size(); ++i )
  {
    objects[i]->scale(L);
  }
}

#endif
