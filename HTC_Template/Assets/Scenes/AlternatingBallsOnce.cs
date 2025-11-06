using UnityEngine;
using UnityEngine.SceneManagement;
using System.Collections;

public class AlternatingBallsOnce : MonoBehaviour
{

    // public XREliteFullLogger logger;  // Make sure it's public or serialized
    public GameObject ball1;
    public GameObject ball2;

    private Renderer rend1, rend2;
    private Material mat1, mat2;
    public float radius = 2f;
    public float duration = 8f;

    void Start()
    {
        rend1 = ball1.GetComponent<Renderer>();
        rend2 = ball2.GetComponent<Renderer>();
        mat1 = rend1.material;
        mat2 = rend2.material;

        // Place balls in front of camera
        Transform cam = Camera.main.transform;
        ball1.transform.position = cam.position + cam.forward * 2 + cam.right * -0.5f;
        ball2.transform.position = cam.position + cam.forward * 2 + cam.right * 0.5f;

        // Initial visibility
        SetAlpha(mat1, 1f); // ball1 visible
        SetAlpha(mat2, 0f); // ball2 transparent

        StartCoroutine(AlternateFor30Seconds());

        if (ball1 == null || ball2 == null)
        {
            Debug.LogError("Ball objects not assigned!");
            return;
        }

        // Camera cam = Camera.main;
        if (cam != null)
        {
            // Generate a random point within a circle (not just the edge) in front of the camera
            Vector2 randomCircle = Random.insideUnitCircle * radius;

            // Offset in camera's right and up directions
            Vector3 offset = cam.transform.right * randomCircle.x + cam.transform.up * randomCircle.y;

            // Set positions in front of the camera at a fixed distance
            Vector3 basePosition = cam.transform.position + cam.transform.forward * 2f;

            ball1.transform.position = basePosition + offset;
            ball2.transform.position = basePosition - offset;
        }
        else
        {
            ball1.transform.position = Random.onUnitSphere * radius;
            ball2.transform.position = Random.onUnitSphere * radius;
        }

        Invoke(nameof(LoadNextScene), duration);
    }

    void LoadNextScene()
    {
        // logger.SaveToCSV();
        SceneManager.LoadScene("EyeTrackerreal");
    }

    IEnumerator AlternateFor30Seconds()
    {
        float elapsed = 0f;

        while (elapsed < 30f)
        {
            // Wait 10 seconds while ball1 is visible
            yield return new WaitForSeconds(10f);
            elapsed += 10f;

            // Fade out ball1, fade in ball2
            yield return StartCoroutine(Fade(mat1, 1f, 0f, 2f));
            yield return StartCoroutine(Fade(mat2, 0f, 1f, 2f));

            // Color transitions on ball2
            mat2.color = new Color(1, 0, 0, 1); yield return new WaitForSeconds(1f); // red
            mat2.color = new Color(1, 1, 0, 1); yield return new WaitForSeconds(1f); // yellow
            mat2.color = new Color(0, 0, 1, 1); yield return new WaitForSeconds(1f); // blue
            elapsed += 5f; // 2s fade + 3s colors

            // Fade out ball2, fade in ball1
            yield return StartCoroutine(Fade(mat2, 1f, 0f, 2f));
            yield return StartCoroutine(Fade(mat1, 0f, 1f, 2f));
            elapsed += 4f;
        }

        Debug.Log("Animation cycle completed.");
    }

    IEnumerator Fade(Material mat, float startAlpha, float endAlpha, float duration)
    {
        float time = 0f;
        while (time < duration)
        {
            float alpha = Mathf.Lerp(startAlpha, endAlpha, time / duration);
            SetAlpha(mat, alpha);
            time += Time.deltaTime;
            yield return null;
        }
        SetAlpha(mat, endAlpha);
    }

    void SetAlpha(Material mat, float alpha)
    {
        Color c = mat.color;
        c.a = alpha;
        mat.color = c;

        // Ensure transparent rendering
        mat.SetFloat("_Mode", 2);
        mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        mat.SetInt("_ZWrite", 0);
        mat.DisableKeyword("_ALPHATEST_ON");
        mat.EnableKeyword("_ALPHABLEND_ON");
        mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        mat.renderQueue = 3000;
    }
}
