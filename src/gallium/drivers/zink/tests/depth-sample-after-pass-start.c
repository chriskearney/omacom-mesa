/* SPDX-License-Identifier: MIT
 * Copyright 2026 Aurora Silicon
 * Standalone surfaceless EGL regression: a depth texture rendered as an
 * attachment is first bound for sampling after the next render pass has
 * already drawn, and that pass's depth buffer is invalidated at its end.
 * cc -O2 -Wall depth-sample-after-pass-start.c -o depth-sample-after-pass-start \
 *    $(pkg-config --cflags --libs egl opengl)
 * Usage: depth-sample-after-pass-start [frames] [late|early]
 *   late : the shadow map is bound to its texture unit only after the main pass has drawn
 *   early: it stays bound to the unit throughout (unused while it is the attachment)
 * Run with zink selected. Exit 0 = pass, 1 = wrong pixels, 77 = skip.
 */
#define GL_GLEXT_PROTOTYPES
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 128

static GLuint
shader(GLenum type, const char *source)
{
   GLuint s = glCreateShader(type);
   glShaderSource(s, 1, &source, NULL);
   glCompileShader(s);
   GLint ok;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096];
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      fprintf(stderr, "%s\n", log);
      exit(2);
   }
   return s;
}

/* Fullscreen triangle at a fixed NDC depth, x limited to [x0, x1]. */
static GLuint
program(const char *fragment)
{
   const char *vertex =
      "#version 330\n"
      "uniform float z; uniform vec2 xr;\n"
      "void main() {\n"
      "  vec2 p[4] = vec2[4](vec2(0,-1), vec2(1,-1), vec2(0,1), vec2(1,1));\n"
      "  vec2 v = p[gl_VertexID];\n"
      "  gl_Position = vec4(mix(xr.x, xr.y, v.x), v.y, z, 1);\n"
      "}\n";
   GLuint vs = shader(GL_VERTEX_SHADER, vertex);
   GLuint fs = shader(GL_FRAGMENT_SHADER, fragment);
   GLuint p = glCreateProgram();
   glAttachShader(p, vs);
   glAttachShader(p, fs);
   glLinkProgram(p);
   GLint ok;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   glDeleteShader(vs);
   glDeleteShader(fs);
   if (!ok)
      exit(2);
   return p;
}

static void
draw(GLuint p, float z, float x0, float x1)
{
   glUseProgram(p);
   glUniform1f(glGetUniformLocation(p, "z"), z);
   glUniform2f(glGetUniformLocation(p, "xr"), x0, x1);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

int
main(int argc, char **argv)
{
   int frames = argc > 1 ? atoi(argv[1]) : 8;
   const bool early = argc > 2 && !strcmp(argv[2], "early");
   if (frames <= 0)
      return 2;

   EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,
                                          EGL_DEFAULT_DISPLAY, NULL);
   if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, NULL, NULL))
      return 77;
   if (!eglBindAPI(EGL_OPENGL_API))
      return 77;
   const EGLint ctx_attrs[] = {
      EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 3,
      EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
      EGL_NONE};
   EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attrs);
   if (ctx == EGL_NO_CONTEXT ||
       !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx))
      return 77;
   printf("renderer=%s\n", glGetString(GL_RENDERER));

   GLuint depth_only = program("#version 330\nvoid main(){}\n");
   GLuint red = program("#version 330\nout vec4 c; void main(){c=vec4(1,0,0,1);}\n");
   /* Green, with blue carrying the sampled shadow-map depth (0.5). */
   GLuint sample = program("#version 330\nuniform sampler2D s; out vec4 c;"
                           "void main(){c=vec4(0,1,texelFetch(s,ivec2(gl_FragCoord.xy),0).r,1);}\n");
   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);

   /* The shadow map, written only as a depth attachment. */
   GLuint shadow, shadow_fb;
   glGenTextures(1, &shadow);
   glBindTexture(GL_TEXTURE_2D, shadow);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, SIZE, SIZE, 0,
                GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glBindTexture(GL_TEXTURE_2D, 0);
   glGenFramebuffers(1, &shadow_fb);
   glBindFramebuffer(GL_FRAMEBUFFER, shadow_fb);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow, 0);
   glDrawBuffer(GL_NONE);
   glReadBuffer(GL_NONE);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return 77;

   /* The main pass: colour texture + its own depth buffer. */
   GLuint color, depth, main_fb;
   glGenTextures(1, &color);
   glBindTexture(GL_TEXTURE_2D, color);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SIZE, SIZE, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
   glBindTexture(GL_TEXTURE_2D, 0);
   glGenRenderbuffers(1, &depth);
   glBindRenderbuffer(GL_RENDERBUFFER, depth);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, SIZE, SIZE);
   glGenFramebuffers(1, &main_fb);
   glBindFramebuffer(GL_FRAMEBUFFER, main_fb);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return 77;

   glViewport(0, 0, SIZE, SIZE);
   glUseProgram(sample);
   glUniform1i(glGetUniformLocation(sample, "s"), 0);
   glActiveTexture(GL_TEXTURE0);

   unsigned failed = 0;
   for (int frame = 0; frame < frames; ++frame) {
      /* 1. Render the shadow map: depth 0.5 everywhere. */
      glBindTexture(GL_TEXTURE_2D, early ? shadow : 0);
      glBindFramebuffer(GL_FRAMEBUFFER, shadow_fb);
      glEnable(GL_DEPTH_TEST);
      glDepthFunc(GL_ALWAYS);
      glClearDepth(1.0);
      glClear(GL_DEPTH_BUFFER_BIT);
      draw(depth_only, 0.0, -1, 1);

      /* 2. Start the main pass and draw an occluder (depth 0.2) on the left. */
      glBindFramebuffer(GL_FRAMEBUFFER, main_fb);
      glClearColor(0, 0, 0, 1);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glDepthFunc(GL_LESS);
      draw(red, -0.6, -1, 0);

      /* 3. Only now bind the shadow map for sampling, and draw a fullscreen
       *    quad at depth 0.5 behind the occluder.
       */
      glBindTexture(GL_TEXTURE_2D, shadow);
      draw(sample, 0.0, -1, 1);

      /* 4. The main pass's depth is not needed afterwards. */
      const GLenum discard = GL_DEPTH_ATTACHMENT;
      glInvalidateFramebuffer(GL_FRAMEBUFFER, 1, &discard);

      unsigned char pixels[SIZE * SIZE * 4];
      glReadPixels(0, 0, SIZE, SIZE, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
      unsigned bad = 0;
      for (int y = 0; y < SIZE; ++y) {
         for (int x = 0; x < SIZE; ++x) {
            const unsigned char *p = pixels + (y * SIZE + x) * 4;
            /* Left half: the occluder. Right half: green + sampled 0.5. */
            const bool left = x < SIZE / 2;
            bad += abs((int)p[0] - (left ? 255 : 0)) > 2;
            bad += abs((int)p[1] - (left ? 0 : 255)) > 2;
            bad += abs((int)p[2] - (left ? 0 : 128)) > 2;
         }
      }
      GLenum error = glGetError();
      failed += bad != 0 || error != GL_NO_ERROR;
      printf("frame=%d bad_rgb_channels=%u error=%#x first=%02x%02x%02x last=%02x%02x%02x\n",
             frame, bad, error, pixels[0], pixels[1], pixels[2],
             pixels[(SIZE * SIZE - 1) * 4], pixels[(SIZE * SIZE - 1) * 4 + 1],
             pixels[(SIZE * SIZE - 1) * 4 + 2]);
   }
   printf("SUMMARY mode=%s frames=%d failed=%u\n", early ? "early" : "late", frames, failed);

   glDeleteFramebuffers(1, &main_fb);
   glDeleteFramebuffers(1, &shadow_fb);
   glDeleteRenderbuffers(1, &depth);
   glDeleteTextures(1, &color);
   glDeleteTextures(1, &shadow);
   glDeleteProgram(depth_only);
   glDeleteProgram(red);
   glDeleteProgram(sample);
   glDeleteVertexArrays(1, &vao);
   eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(dpy, ctx);
   eglTerminate(dpy);
   return failed ? 1 : 0;
}
