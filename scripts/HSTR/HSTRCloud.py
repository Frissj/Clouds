from falcor import *


def render_graph_HSTRCloud():
    graph = RenderGraph("HSTRCloud")
    graph.addPass(createPass("HSTRCloud", {
        "baseSteps": 96,
        "refinementLevel": 1,
        "densityScale": 1.0,
        "residualBlend": 1.0,
        "activeRank": 24,
        "activeThreshold": 0.0,
        "goalFace": 1,
        "traceSpatialOrder": 2,
        "correctionBudget": 262144,
        "correctionFadeFrames": 8,
        "ballisticBudget": 24,
        "nearScatterBudget": 120,
        "diffuseRankBudget": 24,
        "schurWindowRadius": 8,
        "localSourceRadiance": float3(0.0),
        "mixedFidelityThreshold": 0.15,
        "operatorDictionaryTolerance": 1e-4,
    }), "HSTRCloud")
    graph.addPass(createPass("ToneMapper", {
        "autoExposure": False,
        "exposureCompensation": 0.0,
    }), "ToneMapper")
    graph.addPass(createPass("AccumulatePass", {
        "enabled": True,
        "autoReset": True,
    }), "Accumulate")
    graph.addEdge("HSTRCloud.color", "Accumulate.input")
    graph.addEdge("Accumulate.output", "ToneMapper.src")
    graph.markOutput("HSTRCloud.color")
    graph.markOutput("ToneMapper.dst")
    graph.markOutput("HSTRCloud.transportError")
    return graph


HSTRCloud = render_graph_HSTRCloud()
try:
    m.addGraph(HSTRCloud)
except NameError:
    pass
