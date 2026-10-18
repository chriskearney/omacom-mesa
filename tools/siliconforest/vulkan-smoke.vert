// SPDX-License-Identifier: MIT
#version 450
layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 16) out vec4 captured;
void main() {
    gl_Position = vec4(0.0);
    captured = vec4(0.0);
}
