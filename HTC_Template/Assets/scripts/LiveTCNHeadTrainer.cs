using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

/// <summary>
/// Live, on-device online trainer that follows a TCN-style architecture:
///  - Causal, dilated 1D conv blocks (frozen) -> feature stream [T, H]
///  - Trainable softmax head (H -> C) updated online (SGD)
///  - Starts automatically when scene loads; prints weights regularly
/// </summary>
public class LiveTCNHeadTrainer : MonoBehaviour
{
    public enum SceneLabel { Fixation = 0, Saccade = 1, Pursuit = 2 }

    [Header("Gaze sources (assign)")]
    public Transform leftEye;    // your XR LeftEye or equivalent
    public Transform rightEye;   // your XR RightEye or equivalent

    [Header("Scene ground-truth label")]
    public SceneLabel labelForThisScene = SceneLabel.Fixation;

    [Header("Windowing")]
    public int window = 30;     // frames per window (e.g., 30 @120Hz ≈ 0.25s)
    public int stride = 5;      // run an update every N frames

    [Header("Model dims (match your notebook spirit)")]
    public int inF = 6;         // features per frame (LE xyz + RE xyz)
    public int hidden = 16;     // TCN hidden channels (small for headset)
    public int numLevels = 3;   // dilations 1,2,4
    public int kernel = 3;      // causal conv kernel
    public int numClasses = 3;  // Fixation/Saccade/Pursuit

    [Header("Training (head-only)")]
    public float learningRate = 0.05f;
    public float l2 = 1e-4f;

    [Header("Logging")]
    public int printEveryNUpdates = 50;
    public bool printOnDisable = true;

    // ===== runtime buffers =====
    private readonly List<float[]> raw = new List<float[]>(4096);
    private int frameCount = 0, updates = 0;

    // ===== TCN feature extractor (frozen) =====
    private Conv1D[] conv1;  // first conv per level: inCh->hidden
    private Conv1D[] conv2;  // second conv per level: hidden->hidden
    private int[] dilations;

    // ===== trainable head =====
    private float[,] W;  // [hidden, numClasses]
    private float[] b;   // [numClasses]

    // ===== init =====
    void Start()
    {
        if (leftEye == null || rightEye == null)
        {
            Debug.LogError("Assign leftEye/rightEye Transforms.");
            enabled = false; return;
        }

        BuildFrozenTCN();
        W = new float[hidden, numClasses];
        b = new float[numClasses];
        InitSmall(W, b, 0.01f);

        Debug.Log($"[LiveTCNHeadTrainer] Started. window={window}, stride={stride}, levels={numLevels}, hidden={hidden}, kernel={kernel}");
    }

    void Update()
    {
        // read per-frame binocular directions (normalize to be safe)
        Vector3 le = leftEye.forward.normalized;
        Vector3 re = rightEye.forward.normalized;
        raw.Add(new float[] { le.x, le.y, le.z, re.x, re.y, re.z });
        frameCount++;

        // every stride, once we have a full window, do one online update
        if (frameCount >= window && ((frameCount - window) % stride == 0))
        {
            // 1) slice window [T, F]
            float[,] xWin = SliceWindow(raw, frameCount - window, window, inF);

            // 2) forward through TCN -> [T, hidden]
            float[,] hWin = ForwardTCN(xWin);   // causal, dilated blocks

            // 3) we use last frame T-1 for supervision (low latency)
            int T = window;
            float[] hLast = Row(hWin, T - 1);   // [hidden]

            // 4) softmax head forward
            float[] z = HeadLogits(hLast);      // [C]
            float[] p = Softmax(z);             // [C]
            int yTrue = (int)labelForThisScene;

            // 5) SGD step on head only
            HeadSGDStep(hLast, p, yTrue, learningRate, l2);

            // quick feedback
            updates++;
            if (updates % printEveryNUpdates == 0)
            {
                PrintHeadWeights($"update#{updates}");
                PrintRecentPrediction(p, yTrue);
            }
        }
    }

    void OnDisable()
    {
        if (!printOnDisable) return;
        PrintHeadWeights("final");
    }

    // ===== utilities =====

    private float[,] SliceWindow(List<float[]> src, int start, int len, int F)
    {
        var X = new float[len, F];
        for (int t = 0; t < len; t++)
        {
            var v = src[start + t];
            for (int f = 0; f < F; f++) X[t, f] = v[f];
        }
        return X;
    }

    private float[] Row(float[,] M, int r)
    {
        int C = M.GetLength(1);
        var v = new float[C];
        for (int c = 0; c < C; c++) v[c] = M[r, c];
        return v;
    }

    // ====== Head (trainable) ======
    private float[] HeadLogits(float[] h)  // h: [hidden]
    {
        var z = new float[numClasses];
        for (int c = 0; c < numClasses; c++)
        {
            float s = b[c];
            for (int j = 0; j < hidden; j++) s += h[j] * W[j, c];
            z[c] = s;
        }
        return z;
    }

    private void HeadSGDStep(float[] h, float[] p, int yTrue, float lr, float l2)
    {
        for (int c = 0; c < numClasses; c++)
        {
            float err = p[c] - (c == yTrue ? 1f : 0f);
            b[c] -= lr * err;
            for (int j = 0; j < hidden; j++)
            {
                float grad = err * h[j] + l2 * W[j, c];
                W[j, c] -= lr * grad;
            }
        }
    }

    private float[] Softmax(float[] z)
    {
        var p = (float[])z.Clone();
        float m = p[0]; for (int i = 1; i < p.Length; i++) if (p[i] > m) m = p[i];
        float s = 0f; for (int i = 0; i < p.Length; i++) { p[i] = Mathf.Exp(p[i] - m); s += p[i]; }
        float inv = 1f / (s + 1e-9f);
        for (int i = 0; i < p.Length; i++) p[i] *= inv;
        return p;
    }

    private void PrintHeadWeights(string tag)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"=== [{tag}] Softmax head (hidden={hidden} -> classes={numClasses}) ===");
        for (int c = 0; c < numClasses; c++)
        {
            sb.Append($"W[:,{c}] = [");
            for (int j = 0; j < hidden; j++)
            {
                sb.Append(W[j, c].ToString("G6", CultureInfo.InvariantCulture));
                if (j < hidden - 1) sb.Append(", ");
            }
            sb.AppendLine("]");
            sb.AppendLine($"b[{c}] = {b[c].ToString("G6", CultureInfo.InvariantCulture)}");
        }
        Debug.Log(sb.ToString());
    }

    private void PrintRecentPrediction(float[] p, int yTrue)
    {
        int arg = 0; float best = p[0];
        for (int i = 1; i < p.Length; i++) if (p[i] > best) { best = p[i]; arg = i; }
        Debug.Log($"Live pred: class={arg} (p={best:F2}) / true={yTrue}");
    }

    private void InitSmall(float[,] W, float[] b, float scale)
    {
        var rng = new System.Random(123);
        for (int i = 0; i < W.GetLength(0); i++)
            for (int j = 0; j < W.GetLength(1); j++)
                W[i, j] = (float)((rng.NextDouble() * 2 - 1) * scale);
        for (int j = 0; j < b.Length; j++) b[j] = 0f;
    }

    // ====== TCN feature extractor (frozen) ======
    private void BuildFrozenTCN()
    {
        // dilations 1,2,4,... like your notebook
        dilations = new int[numLevels];
        for (int i = 0; i < numLevels; i++) dilations[i] = 1 << i;

        conv1 = new Conv1D[numLevels];
        conv2 = new Conv1D[numLevels];

        // level 0: inF->hidden ; levels >0: hidden->hidden
        int inCh = inF;
        var rng = new System.Random(777);

        for (int i = 0; i < numLevels; i++)
        {
            conv1[i] = new Conv1D(inCh,    hidden, kernel, dilations[i], rng); // causal
            conv2[i] = new Conv1D(hidden,  hidden, kernel, dilations[i], rng);
            inCh = hidden;
        }
    }

    // Forward TCN over a window X[T,F] -> H[T,hidden] (causal, same-time aligned)
    private float[,] ForwardTCN(float[,] X)
    {
        int T = X.GetLength(0);
        // current stream: if first level, treat channels=inF; after that, channels=hidden
        float[,] cur = X; // [T, inCh] initially

        int inCh = X.GetLength(1);

        for (int lvl = 0; lvl < numLevels; lvl++)
        {
            bool first = (lvl == 0);
            int thisIn = first ? inCh : hidden;

            // conv1: thisIn -> hidden
            float[,] y1 = conv1[lvl].Forward(cur, thisIn); // [T, hidden]
            ReLUInPlace(y1);

            // conv2: hidden -> hidden
            float[,] y2 = conv2[lvl].Forward(y1, hidden);
            ReLUInPlace(y2);

            // residual (shape match): if first level and thisIn != hidden, do a 1x1 projection
            float[,] res;
            if (thisIn != hidden)
            {
                // simple projection by channel copy/truncate
                res = ProjectChannels(cur, thisIn, hidden);
            }
            else res = cur;
            cur = AddInPlace(y2, res); // [T, hidden]
        }
        return cur; // [T, hidden]
    }

    private void ReLUInPlace(float[,] A)
    {
        int R = A.GetLength(0), C = A.GetLength(1);
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                if (A[r, c] < 0f) A[r, c] = 0f;
    }

    private float[,] ProjectChannels(float[,] X, int inCh, int outCh)
    {
        int T = X.GetLength(0);
        var Y = new float[T, outCh];
        int copy = Math.Min(inCh, outCh);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < copy; c++)
                Y[t, c] = X[t, c];
        // if outCh > inCh, remaining channels are zeros (OK for residual)
        return Y;
    }

    private float[,] AddInPlace(float[,] A, float[,] B)
    {
        int R = A.GetLength(0), C = A.GetLength(1);
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                A[r, c] += B[r, c];
        return A;
    }

    // ====== minimal causal 1D conv (frozen) ======
    private class Conv1D
    {
        public readonly int inCh, outCh, k, dil;
        private readonly float[,,] W;   // [outCh, inCh, k]
        private readonly float[] b;     // [outCh]

        public Conv1D(int inCh, int outCh, int kernel, int dilation, System.Random rng)
        {
            this.inCh = inCh; this.outCh = outCh; this.k = kernel; this.dil = dilation;
            W = new float[outCh, inCh, k];
            b = new float[outCh];
            float scale = (float)(1.0 / Math.Sqrt(inCh * k));
            for (int o = 0; o < outCh; o++)
            {
                b[o] = 0f;
                for (int i = 0; i < inCh; i++)
                    for (int kk = 0; kk < k; kk++)
                        W[o, i, kk] = (float)((rng.NextDouble() * 2 - 1) * scale);
            }
        }

        // causal SAME: output[t] depends on x[t - d*kk], padding zeros before t<0
        public float[,] Forward(float[,] X, int channelsIn)
        {
            int T = X.GetLength(0);
            var Y = new float[T, outCh];

            for (int t = 0; t < T; t++)
            {
                for (int o = 0; o < outCh; o++)
                {
                    float s = b[o];
                    for (int i = 0; i < channelsIn; i++)
                    {
                        for (int kk = 0; kk < k; kk++)
                        {
                            int tt = t - dil * kk;
                            float xv = (tt >= 0) ? X[tt, i] : 0f; // causal pad
                            s += xv * W[o, i, kk];
                        }
                    }
                    Y[t, o] = s;
                }
            }
            return Y;
        }
    }
}
