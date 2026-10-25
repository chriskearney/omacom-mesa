#version 450
layout(location = 0) out vec4 value;
void main() {
    gl_Position = vec4(float(gl_VertexIndex), 0.0, 0.0, 1.0);
    value = vec4(float(gl_VertexIndex));
}
