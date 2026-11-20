struct VSInput {
    [[vk::location(0)]] float2 inPosition;
    [[vk::location(1)]] float3 inColor;
};

struct VertexOutput {
    float4 sv_position : SV_Position;
    float3 color;
};

[shader("vertex")]
VertexOutput main(VSInput input) {
    VertexOutput output;
    output.sv_position = float4(input.inPosition, 0.0, 1.0);
    output.color = input.inColor;
    return output;
}