using UnityEngine;
using UnityEngine.SceneManagement;
using System.Collections;

public class AlternatingBallsOnce : MonoBehaviour
{
    public GameObject ball1;
    public GameObject ball2;

    private Material mat1, mat2;
    public float radius = 2f;
    public float duration = 8f;

    private static readonly int ColorProp = Shader.PropertyToID("_Color");
    private static readonly int ModeProp = Shader.PropertyToID("_Mode");
    private static readonly int SrcBlendProp = Shader.PropertyToID("_SrcBlend");
    private static readonly int DstBlendProp = Shader.PropertyToID("_DstBlend");
    private static readonly int ZWriteProp = Shader.PropertyToID("_ZWrite");

    // Add color sequence and index
    // Color sequence: GREEN -> YELLOW -> RED (alpha preserved by SetColor)
    private Color[] colorSequence = new Color[] {
        new Color(0, 1, 0, 1), // green
        new Color(1, 1, 0, 1), // yellow
        new Color(1, 0, 0, 1)  // red
    };
    private int colorIndexBall1 = 0;
    private int colorIndexBall2 = 0;

    void Start()
    {
        if (ball1 == null || ball2 == null)
        {
            Debug.LogError("Ball objects not assigned!");
            return;
        }

        mat1 = ball1.GetComponent<Renderer>().material;
        mat2 = ball2.GetComponent<Renderer>().material;

        Transform cam = Camera.main?.transform;
        if (cam != null)
        {
            Vector3 basePosition = cam.position + cam.forward * 2f;
            Vector2 randomCircle = Random.insideUnitCircle * radius;
            Vector3 offset = cam.right * randomCircle.x + cam.up * randomCircle.y;
            ball1.transform.position = basePosition + offset;
            ball2.transform.position = basePosition - offset;
        }
        else
        {
            Vector3 basePosition = Vector3.forward * 2f;
            Vector2 randomCircle = Random.insideUnitCircle * radius;
            Vector3 offset = Vector3.right * randomCircle.x + Vector3.up * randomCircle.y;
            ball1.transform.position = basePosition + offset;
            ball2.transform.position = basePosition - offset;
        }

        SetAlpha(mat1, 1f);
        SetAlpha(mat2, 0f);

        // Set initial colors
        SetColor(mat1, colorSequence[colorIndexBall1]);
        SetColor(mat2, colorSequence[colorIndexBall2]);

        // Start alternating loop: each visible-phase lasts visibleDuration (10s) with colors cycling,
        // then transparent-phase lasts transparentDuration (10s). The loop runs until scene unload.
        StartCoroutine(AlternateFor30Seconds());
        Invoke(nameof(LoadNextScene), duration);

    }

    void LoadNextScene()
    {
        SceneManager.LoadScene("EyeTrackerreal");
    }

    IEnumerator AlternateFor30Seconds()
    {
        float elapsed = 0f;

        while (elapsed < 30f)
        {
            yield return new WaitForSeconds(10f);
            elapsed += 10f;

            // Fade out ball1, fade in ball2
            yield return Fade(mat1, 1f, 0f, 2f);
            yield return Fade(mat2, 0f, 1f, 2f);

            // Change ball2 color in sequence
            colorIndexBall2 = (colorIndexBall2 + 1) % colorSequence.Length;
            SetColor(mat2, colorSequence[colorIndexBall2]);

            yield return new WaitForSeconds(1f);

            colorIndexBall2 = (colorIndexBall2 + 1) % colorSequence.Length;
            SetColor(mat2, colorSequence[colorIndexBall2]);

            yield return new WaitForSeconds(1f);

            colorIndexBall2 = (colorIndexBall2 + 1) % colorSequence.Length;
            SetColor(mat2, colorSequence[colorIndexBall2]);

            yield return new WaitForSeconds(1f);
            elapsed += 5f;

            // Fade out ball2, fade in ball1
            yield return Fade(mat2, 1f, 0f, 2f);
            yield return Fade(mat1, 0f, 1f, 2f);

            // Change ball1 color in sequence
            colorIndexBall1 = (colorIndexBall1 + 1) % colorSequence.Length;
            SetColor(mat1, colorSequence[colorIndexBall1]);

            yield return new WaitForSeconds(1f);

            colorIndexBall1 = (colorIndexBall1 + 1) % colorSequence.Length;
            SetColor(mat1, colorSequence[colorIndexBall1]);

            yield return new WaitForSeconds(1f);

            colorIndexBall1 = (colorIndexBall1 + 1) % colorSequence.Length;
            SetColor(mat1, colorSequence[colorIndexBall1]);

            yield return new WaitForSeconds(1f);
            elapsed += 4f;
        }
    }

    IEnumerator Fade(Material mat, float startAlpha, float endAlpha, float duration)
    {
        float time = 0f;
        Color c = mat.GetColor(ColorProp);
        while (time < duration)
        {
            c.a = Mathf.Lerp(startAlpha, endAlpha, time / duration);
            mat.SetColor(ColorProp, c);
            time += Time.deltaTime;
            yield return null;
        }
        c.a = endAlpha;
        mat.SetColor(ColorProp, c);
    }

    void SetAlpha(Material mat, float alpha)
    {
        Color c = mat.GetColor(ColorProp);
        c.a = alpha;
        mat.SetColor(ColorProp, c);

        mat.SetFloat(ModeProp, 2);
        mat.SetInt(SrcBlendProp, (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        mat.SetInt(DstBlendProp, (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        mat.SetInt(ZWriteProp, 0);
        mat.DisableKeyword("_ALPHATEST_ON");
        mat.EnableKeyword("_ALPHABLEND_ON");
        mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        mat.renderQueue = 3000;
    }

    // Overload for Color - preserve current alpha when changing color
    void SetColor(Material mat, Color color)
    {
        Color cur = mat.GetColor(ColorProp);
        color.a = cur.a;
        mat.SetColor(ColorProp, color);
    }
}
