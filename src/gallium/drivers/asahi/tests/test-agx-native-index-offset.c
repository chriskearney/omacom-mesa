/* SPDX-License-Identifier: MIT */
#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line=%d GL=%x EGL=%x\n", __LINE__, glGetError(), eglGetError()); exit(2); } } while (0)
static GLuint
shader(GLenum type, const char *source)
{
   GLuint s = glCreateShader(type);
   glShaderSource(s, 1, &source, NULL);
   glCompileShader(s);
   GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   CHECK(ok);
   return s;
}
int
main(void)
{
   setvbuf(stdout, NULL, _IONBF, 0);
   EGLDisplay d = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, NULL, NULL);
   CHECK(eglInitialize(d, NULL, NULL));
   CHECK(eglBindAPI(EGL_OPENGL_ES_API));
   const EGLint attributes[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE};
   EGLContext c = eglCreateContext(d, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, attributes);
   CHECK(c != EGL_NO_CONTEXT);
   CHECK(eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, c));
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   printf("renderer=%s\n", renderer);
   CHECK(renderer && strstr(renderer, "M3 Pro") && !strstr(renderer, "zink"));
   const char *vs = "#version 300 es\nlayout(location=0) in vec2 position;void main(){gl_Position=vec4(position,0,1);}";
   const char *fs = "#version 300 es\nprecision highp float;out vec4 color;void main(){color=vec4(1);}";
   GLuint program = glCreateProgram();
   glAttachShader(program, shader(GL_VERTEX_SHADER, vs));
   glAttachShader(program, shader(GL_FRAGMENT_SHADER, fs));
   glLinkProgram(program); GLint linked; glGetProgramiv(program, GL_LINK_STATUS, &linked); CHECK(linked);
   glUseProgram(program);
   GLuint fb, tex, vao, vertex, index;
   glGenFramebuffers(1, &fb); glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 256, 256);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
   CHECK(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
   glGenVertexArrays(1, &vao); glBindVertexArray(vao);
   glGenBuffers(1, &vertex); glBindBuffer(GL_ARRAY_BUFFER, vertex);
   const float positions[] = {-1,-1, 1,-1, -1,1, 1,1};
   glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL); glEnableVertexAttribArray(0);
   glGenBuffers(1, &index); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index);
   glViewport(0, 0, 256, 256); glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND); glDisable(GL_CULL_FACE);
   const unsigned points[4][2] = {{32,48}, {224,64}, {64,224}, {208,208}};
   unsigned cases = 0, failures = 0;
   for (unsigned size = 1; size <= 4; size *= 2) {
      const unsigned offsets8[] = {0,1,3}, offsets16[] = {0,2,4}, offsets32[] = {0,4};
      const unsigned *offsets = size == 1 ? offsets8 : size == 2 ? offsets16 : offsets32;
      const unsigned offset_count = size == 4 ? 2 : 3;
      for (unsigned at = 0; at < offset_count; ++at)
      for (unsigned triangles = 0; triangles < 2; ++triangles)
      for (unsigned restart = 0; restart < 2; ++restart) {
         const unsigned ids[] = {0,1,2,2,1,3}, strip[] = {0,1,2,3};
         const unsigned *values = triangles ? ids : strip;
         const unsigned count = triangles ? 6 : 4;
         const unsigned offset = offsets[at];
         uint8_t data[32]; memset(data, 0xff, sizeof(data));
         for (unsigned i = 0; i < count; ++i) {
            if (size == 1) data[offset+i] = values[i];
            else if (size == 2) { uint16_t value = values[i]; memcpy(data+offset+2*i, &value, 2); }
            else { uint32_t value = values[i]; memcpy(data+offset+4*i, &value, 4); }
         }
         glBufferData(GL_ELEMENT_ARRAY_BUFFER, offset+count*size, data, GL_STATIC_DRAW);
         if (restart) glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX); else glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
         glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
         glDrawElements(triangles ? GL_TRIANGLES : GL_TRIANGLE_STRIP, count,
                        size == 1 ? GL_UNSIGNED_BYTE : size == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT,
                        (void *)(uintptr_t)offset);
         glFinish(); CHECK(glGetError() == GL_NO_ERROR);
         unsigned bad = 0;
         for (unsigned i = 0; i < 4; ++i) {
            uint8_t pixel[4]; glReadPixels(points[i][0], points[i][1], 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
            CHECK(glGetError() == GL_NO_ERROR);
            bad += pixel[0] != 255 || pixel[1] != 255 || pixel[2] != 255 || pixel[3] != 255;
         }
         failures += bad != 0; ++cases;
         printf("size=%u offset=%u triangles=%u restart=%u bad=%u\n", size, offset, triangles, restart, bad);
      }
   }
   printf("native-index-offset cases=%u failures=%u\n", cases, failures);
   return failures ? 1 : 0;
}
