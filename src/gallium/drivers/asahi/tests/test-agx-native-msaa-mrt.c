/* Copyright 2026 Chris Kearney
 * SPDX-License-Identifier: MIT */
#define GL_GLEXT_PROTOTYPES
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

#define W 64
#define H 64
#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "line=%d GL=%x\n", __LINE__, glGetError());           \
         return 2;                                                             \
      }                                                                        \
   } while (0)

static unsigned
check_target(GLuint source, GLuint resolved, unsigned rt, unsigned phase)
{
   glBindFramebuffer(GL_READ_FRAMEBUFFER, source);
   glReadBuffer(GL_COLOR_ATTACHMENT0 + rt);
   glBindFramebuffer(GL_DRAW_FRAMEBUFFER, resolved);
   glBlitFramebuffer(0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT, GL_NEAREST);
   glBindFramebuffer(GL_READ_FRAMEBUFFER, resolved);
   glReadBuffer(GL_COLOR_ATTACHMENT0);
   unsigned char pixels[W * H * 4];
   glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
   unsigned bad = 0;
   for (unsigned y = 0; y < H; ++y) {
      for (unsigned x = 0; x < W; ++x) {
         const bool blue = phase == 1 && rt == 0 && x < W / 2;
         unsigned char expected[4] = {
            rt == 0 && !blue ? 255 : 0,
            rt == 1 ? 255 : 0,
            blue ? 255 : 0,
            255,
         };
         if (phase == 2) {
            expected[0] = 128;
            expected[1] = rt ? 128 : 0;
            expected[2] = rt ? 0 : 128;
         } else if (phase == 3) {
            expected[0] = rt ? 128 : 0;
            expected[1] = rt ? 0 : 128;
            expected[2] = 128;
         }
         const unsigned tolerance = phase >= 2 ? 1 : 0;
         bool different = false;
         for (unsigned c = 0; c < 4; ++c)
            different |=
               abs((int)pixels[(y * W + x) * 4 + c] - expected[c]) > tolerance;
         bad += different;
      }
   }
   const GLenum error = glGetError();
   const char *names[] = {"clear-store", "reload-partial-clear",
                          "flush-half-samples", "reload-complementary-samples"};
   printf("case=%s rt=%u bad=%u GL=%x\n", names[phase], rt, bad, error);
   return bad != 0 || error != GL_NO_ERROR;
}

int
main(int argc, char **argv)
{
   CHECK(argc == 2);
   const unsigned samples = strtoul(argv[1], NULL, 10);
   CHECK(samples == 1 || samples == 2 || samples == 4);
   setvbuf(stdout, NULL, _IONBF, 0);
   EGLDisplay d =
      eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, NULL, NULL);
   CHECK(eglInitialize(d, NULL, NULL));
   CHECK(eglBindAPI(EGL_OPENGL_ES_API));
   EGLint attrs[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE};
   EGLContext c = eglCreateContext(d, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, attrs);
   CHECK(c != EGL_NO_CONTEXT);
   CHECK(eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, c));
   GLint max_rt, max_samples;
   glGetIntegerv(GL_MAX_DRAW_BUFFERS, &max_rt);
   glGetIntegerv(GL_MAX_SAMPLES, &max_samples);
   CHECK(max_rt >= 2 && max_samples >= (int)samples);
   printf("renderer=%s maxrt=%d maxsamples=%d samples=%u\n",
          glGetString(GL_RENDERER), max_rt, max_samples, samples);

   GLuint source, rb[2], resolved, texture;
   glGenFramebuffers(1, &source);
   glBindFramebuffer(GL_FRAMEBUFFER, source);
   glGenRenderbuffers(2, rb);
   for (unsigned i = 0; i < 2; ++i) {
      glBindRenderbuffer(GL_RENDERBUFFER, rb[i]);
      if (samples == 1)
         glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, W, H);
      else
         glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA8, W,
                                          H);
      GLint actual_samples;
      glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_SAMPLES,
                                   &actual_samples);
      CHECK(actual_samples == (samples == 1 ? 0 : (int)samples));
      printf("rt=%u storage_samples=%d\n", i, actual_samples);
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i,
                                GL_RENDERBUFFER, rb[i]);
   }
   GLenum bufs[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
   glDrawBuffers(2, bufs);
   CHECK(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
   const float red[] = {1, 0, 0, 1}, green[] = {0, 1, 0, 1};
   glClearBufferfv(GL_COLOR, 0, red);
   glClearBufferfv(GL_COLOR, 1, green);
   CHECK(glGetError() == GL_NO_ERROR);
   puts("clear=accepted flush=next");
   glFinish();
   CHECK(glGetError() == GL_NO_ERROR);
   puts("flush=completed");

   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, H);
   glGenFramebuffers(1, &resolved);
   glBindFramebuffer(GL_DRAW_FRAMEBUFFER, resolved);
   glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                          GL_TEXTURE_2D, texture, 0);
   CHECK(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) ==
         GL_FRAMEBUFFER_COMPLETE);
   unsigned failures = check_target(source, resolved, 0, 0) +
                       check_target(source, resolved, 1, 0);

   /* Resolving an attachment must retain its samples. Reload both targets and
    * clear only half of RT0, preserving RT1 and the other half of RT0. */
   glBindFramebuffer(GL_FRAMEBUFFER, source);
   glEnable(GL_SCISSOR_TEST);
   glScissor(0, 0, W / 2, H);
   const float blue[] = {0, 0, 1, 1};
   glClearBufferfv(GL_COLOR, 0, blue);
   glDisable(GL_SCISSOR_TEST);
   CHECK(glGetError() == GL_NO_ERROR);
   glFinish();
   CHECK(glGetError() == GL_NO_ERROR);
   failures += check_target(source, resolved, 0, 1) +
               check_target(source, resolved, 1, 1);
   if (samples > 1) {
      const char *vs =
         "#version 300 es\nvoid main(){gl_Position=vec4("
         "(gl_VertexID==1?3.:-1.),(gl_VertexID==2?3.:-1.),0.,1.);}";
      const char *fs =
         "#version 300 es\nprecision highp float;"
         "uniform vec4 tint0,tint1;layout(location=0) out vec4 c0;"
         "layout(location=1) out vec4 c1;void main(){c0=tint0;c1=tint1;}";
      GLuint v = glCreateShader(GL_VERTEX_SHADER);
      GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
      glShaderSource(v, 1, &vs, NULL);
      glCompileShader(v);
      glShaderSource(f, 1, &fs, NULL);
      glCompileShader(f);
      GLuint program = glCreateProgram();
      glAttachShader(program, v);
      glAttachShader(program, f);
      glLinkProgram(program);
      GLint linked;
      glGetProgramiv(program, GL_LINK_STATUS, &linked);
      CHECK(linked);
      glUseProgram(program);
      GLint tint0 = glGetUniformLocation(program, "tint0");
      GLint tint1 = glGetUniformLocation(program, "tint1");
      CHECK(tint0 >= 0 && tint1 >= 0);
      glBindFramebuffer(GL_FRAMEBUFFER, source);
      glClearBufferfv(GL_COLOR, 0, red);
      glClearBufferfv(GL_COLOR, 1, green);
      glViewport(0, 0, W, H);
      glEnable(GL_SAMPLE_COVERAGE);
      glSampleCoverage(0.5, GL_FALSE);
      glUniform4f(tint0, 0, 0, 1, 1);
      glUniform4f(tint1, 1, 0, 0, 1);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      glFinish();
      CHECK(glGetError() == GL_NO_ERROR);
      failures += check_target(source, resolved, 0, 2) +
                  check_target(source, resolved, 1, 2);

      /* The next draw changes only the complementary samples after a flush
       * and resolve. Averaged reloads would lose the original sample colors. */
      glBindFramebuffer(GL_FRAMEBUFFER, source);
      glSampleCoverage(0.5, GL_TRUE);
      glUniform4f(tint0, 0, 1, 0, 1);
      glUniform4f(tint1, 0, 0, 1, 1);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      glFinish();
      CHECK(glGetError() == GL_NO_ERROR);
      glDisable(GL_SAMPLE_COVERAGE);
      failures += check_target(source, resolved, 0, 3) +
                  check_target(source, resolved, 1, 3);
   }
   printf("samples=%u failures=%u\n", samples, failures);
   return failures ? 1 : 0;
}
