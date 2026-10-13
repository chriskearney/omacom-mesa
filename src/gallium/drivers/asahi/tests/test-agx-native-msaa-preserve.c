/* SPDX-License-Identifier: MIT
 * Copyright 2026 Chris Kearney
 */
#define GL_GLEXT_PROTOTYPES
#include <stdio.h>
#include <stdlib.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#define W 100
#define H 50
#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "line=%d GL=%x\n", __LINE__, glGetError());           \
         return 2;                                                             \
      }                                                                        \
   } while (0)
static GLuint program;
static GLint color;
static void
draw(float r, float g, float b)
{
   glUniform4f(color, r, g, b, 1);
   glDrawArrays(GL_TRIANGLES, 0, 3);
}
static unsigned
check_pixels(const char *name, int samples, int mode)
{
   unsigned char pixels[W * H * 4];
   glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
   unsigned bad = 0;
   for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
         int expected[4] = {255, 0, 0, 255}, tolerance = 0;
         if (mode == 1) {
            expected[0] = 128;
            expected[2] = 128;
            tolerance = 1;
         }
         if (mode == 2) {
            expected[0] = 128;
            expected[1] = 128;
            tolerance = 1;
         }
         if (mode == 3 && x >= W / 2) {
            expected[0] = 0;
            expected[1] = 255;
         }
         if (mode == 4 && (x < 25 || x >= 75)) {
            expected[0] = 0;
            expected[2] = 255;
         }
         int different = 0;
         for (int c = 0; c < 4; c++)
            different |=
               abs((int)pixels[(y * W + x) * 4 + c] - expected[c]) > tolerance;
         bad += different;
      }
   GLenum error = glGetError();
   printf(
      "case=%s samples=%d bad=%u first=%u,%u,%u,%u center=%u,%u,%u,%u GL=%x\n",
      name, samples, bad, pixels[0], pixels[1], pixels[2], pixels[3],
      pixels[(W / 2) * 4], pixels[(W / 2) * 4 + 1], pixels[(W / 2) * 4 + 2],
      pixels[(W / 2) * 4 + 3], error);
   return bad + (error != GL_NO_ERROR);
}
int
main(void)
{
   EGLDisplay d =
      eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, NULL, NULL);
   CHECK(eglInitialize(d, NULL, NULL));
   CHECK(eglBindAPI(EGL_OPENGL_ES_API));
   EGLint attr[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE};
   EGLContext ctx =
      eglCreateContext(d, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, attr);
   CHECK(ctx != EGL_NO_CONTEXT);
   CHECK(eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx));
   printf("renderer=%s\n", glGetString(GL_RENDERER));
   const char *vs =
      "#version 300 es\nvoid main(){gl_Position=vec4((gl_VertexID==1?3.:-1.),(gl_VertexID==2?3.:-1.),0.,1.);}";
   const char *fs =
      "#version 300 es\nprecision highp float;uniform vec4 tint;out vec4 c;void main(){c=tint;}";
   GLuint v = glCreateShader(GL_VERTEX_SHADER),
          f = glCreateShader(GL_FRAGMENT_SHADER);
   glShaderSource(v, 1, &vs, NULL);
   glCompileShader(v);
   glShaderSource(f, 1, &fs, NULL);
   glCompileShader(f);
   program = glCreateProgram();
   glAttachShader(program, v);
   glAttachShader(program, f);
   glLinkProgram(program);
   GLint linked;
   glGetProgramiv(program, GL_LINK_STATUS, &linked);
   CHECK(linked);
   glUseProgram(program);
   color = glGetUniformLocation(program, "tint");
   CHECK(color >= 0);
   glDisable(GL_DITHER);
   const char *names[] = {"direct",
                          "half-coverage",
                          "flush-half-coverage",
                          "flush-complementary",
                          "resolve-reload-scissor",
                          "partial-rectangle",
                          "flush-full",
                          "complementary-no-flush",
                          "resolve-complementary"};
   unsigned failures = 0;
   for (int samples = 2; samples <= 4; samples *= 2)
      for (int scenario = 0; scenario < 9; scenario++) {
         GLuint ms, ss, rb, texture;
         glGenFramebuffers(1, &ms);
         glBindFramebuffer(GL_FRAMEBUFFER, ms);
         glGenRenderbuffers(1, &rb);
         glBindRenderbuffer(GL_RENDERBUFFER, rb);
         glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA8, W,
                                          H);
         glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_RENDERBUFFER, rb);
         CHECK(glCheckFramebufferStatus(GL_FRAMEBUFFER) ==
               GL_FRAMEBUFFER_COMPLETE);
         glGenTextures(1, &texture);
         glBindTexture(GL_TEXTURE_2D, texture);
         glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, H);
         glGenFramebuffers(1, &ss);
         glBindFramebuffer(GL_FRAMEBUFFER, ss);
         glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                GL_TEXTURE_2D, texture, 0);
         CHECK(glCheckFramebufferStatus(GL_FRAMEBUFFER) ==
               GL_FRAMEBUFFER_COMPLETE);
         glClearColor(0, 0, 1, 1);
         glClear(GL_COLOR_BUFFER_BIT);
         glBindFramebuffer(GL_FRAMEBUFFER, ms);
         glViewport(0, 0, W, H);
         glClearColor(0, 0, 1, 1);
         glClear(GL_COLOR_BUFFER_BIT);
         if ((scenario >= 1 && scenario <= 3) || scenario == 7 ||
             scenario == 8) {
            glEnable(GL_SAMPLE_COVERAGE);
            glSampleCoverage(.5f, GL_FALSE);
         }
         draw(1, 0, 0);
         if (scenario == 2 || scenario == 3 || scenario == 6)
            glFinish();
         if (scenario == 3 || scenario == 7) {
            glSampleCoverage(.5f, GL_TRUE);
            draw(0, 1, 0);
         }
         glDisable(GL_SAMPLE_COVERAGE);
         glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ss);
         if (scenario == 5)
            glBlitFramebuffer(25, 0, 75, H, 25, 0, 75, H, GL_COLOR_BUFFER_BIT,
                              GL_NEAREST);
         else
            glBlitFramebuffer(0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT,
                              GL_NEAREST);
         glBindFramebuffer(GL_FRAMEBUFFER, ss);
         CHECK(glGetError() == GL_NO_ERROR);
         int mode = scenario == 1 || scenario == 2 || scenario == 8 ? 1
                    : scenario == 3 || scenario == 7                ? 2
                    : scenario == 5                                 ? 4
                                                                    : 0;
         failures += check_pixels(names[scenario], samples, mode) != 0;
         CHECK(glGetError() == GL_NO_ERROR);
         if (scenario == 4 || scenario == 8) {
            glBindFramebuffer(GL_FRAMEBUFFER, ms);
            if (scenario == 8) {
               glEnable(GL_SAMPLE_COVERAGE);
               glSampleCoverage(.5f, GL_TRUE);
            } else {
               glEnable(GL_SCISSOR_TEST);
               glScissor(W / 2, 0, W / 2, H);
            }
            draw(0, 1, 0);
            glDisable(GL_SCISSOR_TEST);
            glDisable(GL_SAMPLE_COVERAGE);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ss);
            glBlitFramebuffer(0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT,
                              GL_NEAREST);
            glBindFramebuffer(GL_FRAMEBUFFER, ss);
            CHECK(glGetError() == GL_NO_ERROR);
            failures +=
               check_pixels(scenario == 8 ? "resolve-complementary-second"
                                          : "resolve-reload-scissor-second",
                            samples, scenario == 8 ? 2 : 3) != 0;
            CHECK(glGetError() == GL_NO_ERROR);
         }
         glDeleteFramebuffers(1, &ms);
         glDeleteFramebuffers(1, &ss);
         glDeleteRenderbuffers(1, &rb);
         glDeleteTextures(1, &texture);
      }
   printf("failures=%u\n", failures);
   return failures ? 1 : 0;
}
