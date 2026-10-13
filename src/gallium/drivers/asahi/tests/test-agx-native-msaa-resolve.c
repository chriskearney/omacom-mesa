/* SPDX-License-Identifier: MIT
 * Copyright 2026 Chris Kearney
 * Pixel checks for native multisample clear, draw and color resolve.
 */
#define GL_GLEXT_PROTOTYPES
#include <stdio.h>
#include <stdlib.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "line=%d GL=%x\n", __LINE__, glGetError());           \
         return 2;                                                             \
      }                                                                        \
   } while (0)
int
main(void)
{
   EGLDisplay d =
      eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, NULL, NULL);
   CHECK(eglInitialize(d, NULL, NULL));
   CHECK(eglBindAPI(EGL_OPENGL_ES_API));
   EGLint attrs[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE};
   EGLContext c = eglCreateContext(d, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, attrs);
   CHECK(c != EGL_NO_CONTEXT);
   CHECK(eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, c));
   printf("renderer=%s\n", glGetString(GL_RENDERER));
   const char *vs =
      "#version 300 es\nvoid main(){gl_Position=vec4((gl_VertexID==1?3.:-1.),(gl_VertexID==2?3.:-1.),0.,1.);}";
   const char *fs =
      "#version 300 es\nprecision highp float;out vec4 color;void main(){color=vec4(1.,0.,0.,1.);}";
   GLuint p = glCreateProgram(), v = glCreateShader(GL_VERTEX_SHADER),
          f = glCreateShader(GL_FRAGMENT_SHADER);
   glShaderSource(v, 1, &vs, NULL);
   glCompileShader(v);
   glShaderSource(f, 1, &fs, NULL);
   glCompileShader(f);
   glAttachShader(p, v);
   glAttachShader(p, f);
   glLinkProgram(p);
   GLint ok;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   CHECK(ok);
   glUseProgram(p);
   unsigned failures = 0;
   for (int samples = 2; samples <= 4; samples *= 2)
      for (int draw = 0; draw < 2; draw++)
         for (int with_depth = 0; with_depth < 2; with_depth++) {
            GLuint ms, ss, rb, ds, t;
            glGenFramebuffers(1, &ms);
            glBindFramebuffer(GL_FRAMEBUFFER, ms);
            glGenRenderbuffers(1, &rb);
            glBindRenderbuffer(GL_RENDERBUFFER, rb);
            glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA8,
                                             100, 50);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                      GL_RENDERBUFFER, rb);
            ds = 0;
            if (with_depth) {
               glGenRenderbuffers(1, &ds);
               glBindRenderbuffer(GL_RENDERBUFFER, ds);
               glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples,
                                                GL_DEPTH24_STENCIL8, 100, 50);
               glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                                         GL_DEPTH_STENCIL_ATTACHMENT,
                                         GL_RENDERBUFFER, ds);
            }
            CHECK(glCheckFramebufferStatus(GL_FRAMEBUFFER) ==
                  GL_FRAMEBUFFER_COMPLETE);
            glGenTextures(1, &t);
            glBindTexture(GL_TEXTURE_2D, t);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 100, 50);
            glGenFramebuffers(1, &ss);
            glBindFramebuffer(GL_FRAMEBUFFER, ss);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, t, 0);
            CHECK(glCheckFramebufferStatus(GL_FRAMEBUFFER) ==
                  GL_FRAMEBUFFER_COMPLETE);
            glBindFramebuffer(GL_FRAMEBUFFER, ms);
            glViewport(0, 0, 100, 50);
            glClearColor(draw ? 0 : 1, 0, 0, draw ? 0 : 1);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT |
                    GL_STENCIL_BUFFER_BIT);
            if (draw)
               glDrawArrays(GL_TRIANGLES, 0, 3);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ss);
            glBlitFramebuffer(0, 0, 100, 50, 0, 0, 100, 50, GL_COLOR_BUFFER_BIT,
                              GL_NEAREST);
            glBindFramebuffer(GL_FRAMEBUFFER, ss);
            unsigned char pixels[100 * 50 * 4];
            glReadPixels(0, 0, 100, 50, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
            CHECK(glGetError() == GL_NO_ERROR);
            unsigned bad = 0;
            for (int i = 0; i < 5000; i++)
               bad += pixels[i * 4] != 255 || pixels[i * 4 + 1] != 0 ||
                      pixels[i * 4 + 2] != 0 || pixels[i * 4 + 3] != 255;
            failures += bad != 0;
            printf("samples=%d draw=%d depth=%d bad=%u first=%u,%u,%u,%u\n",
                   samples, draw, with_depth, bad, pixels[0], pixels[1],
                   pixels[2], pixels[3]);
            glDeleteFramebuffers(1, &ms);
            glDeleteFramebuffers(1, &ss);
            glDeleteTextures(1, &t);
            glDeleteRenderbuffers(1, &rb);
            glDeleteRenderbuffers(1, &ds);
         }
   return failures ? 1 : 0;
}
