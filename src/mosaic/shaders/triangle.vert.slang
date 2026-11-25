struct VSInput {
    [[vk::location(0)]] float2 inPosition;
    [[vk::location(1)]] float3 inColor;
};

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};
ConstantBuffer<UniformBuffer> ubo;

struct VertexOutput {
    float4 pos : SV_Position;
    float3 color;
};

[shader("vertex")]
VertexOutput main(VSInput input) {
    VertexOutput output;
    output.pos = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPosition, 0.0, 1.0))));
    output.color = input.inColor;
    return output;
}