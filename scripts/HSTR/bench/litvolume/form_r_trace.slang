// Form R gate R0 (LIT_VOLUME.md section 6c): the inline ray queries a march would issue to cross empty space between the shared
// occupancy hulls. One thread a lattice ray. The hulls of overlapping instances (both layers share the slab) are closed, so the
// union's inside is a count: +1 at a front face, -1 at a back face. Each query takes the next face of any facing; a ray is in
// cloud range while the count is positive. Every face crossing costs one query - the march could not skip them, since a jump
// is only exact from outside every hull.

struct RayRec
{
    float3 origin;
    float tmax;
    float3 dir;
    float tmin;
};

RaytracingAccelerationStructure g_tlas;
StructuredBuffer<RayRec> g_rays;
StructuredBuffer<uint> g_order; // thread -> ray: identity keeps the 8 x 8 lattice tiles together, a permutation shuffles them
RWStructuredBuffer<uint4> g_out; // per ray: queries, union entries, first hit was a back face, asuint(first union entry t)

uniform uint g_count;
uniform uint g_maxQueries; // 1: only the first entry (what rasterising the hulls would give each ray)

[shader("compute")]
[numthreads(64, 1, 1)]
void traceRuns(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= g_count)
        return;
    const uint i = g_order[tid.x];
    const RayRec r = g_rays[i];
    RayDesc ray;
    ray.Origin = r.origin;
    ray.Direction = r.dir;
    ray.TMin = r.tmin;
    ray.TMax = r.tmax;
    int inside = 0;
    uint queries = 0, entries = 0, firstBack = 0;
    float firstT = -1.0;
    while (queries < g_maxQueries && ray.TMin < ray.TMax)
    {
        RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
        q.TraceRayInline(g_tlas, RAY_FLAG_NONE, 0xFF, ray);
        while (q.Proceed())
        {
        }
        queries++;
        if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
            break;
        const float t = q.CommittedRayT();
        if (q.CommittedTriangleFrontFace())
        {
            if (inside == 0)
            {
                entries++;
                if (firstT < 0.0)
                    firstT = t;
            }
            inside++;
        }
        else
        {
            if (queries == 1)
                firstBack = 1;
            inside = max(inside - 1, 0);
        }
        ray.TMin = t + max(1e-3, t * 2e-6); // past this face; faces of different instances are never coplanar in practice
    }
    g_out[i] = uint4(queries, entries, firstBack, asuint(firstT));
}
