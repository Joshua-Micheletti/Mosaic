struct VertexOutput {
    float4 sv_position : SV_Position;
    float3 color;
};

[shader("fragment")]
float4 main(VertexOutput inVert) : SV_Target
{
    float3 color = inVert.color;
    return float4(color, 1.0);
}