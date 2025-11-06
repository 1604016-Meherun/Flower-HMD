using UnityEngine;
using UnityEngine.SceneManagement;
using System.Collections;

public class SmoothSphereMovement : MonoBehaviour
{
    [SerializeField] private float radius = 1.5f;
    [SerializeField] private SmoothPursuitLogger sslogger;
    [SerializeField] private float speed = 0.5f;
    [SerializeField] private float duration = 30f;

    private Vector3 noiseOffsets;
    private Vector3 basePosition;

    private void Start()
    {
        basePosition = transform.position;
        noiseOffsets = new Vector3(
            Random.Range(0f, 100f),
            Random.Range(100f, 200f),
            Random.Range(200f, 300f)
        );
        StartCoroutine(RunSceneForDuration());
    }

    private void Update()
    {
        float t = Time.time * speed;
        float x = (Mathf.PerlinNoise(t + noiseOffsets.x, 0f) - 0.5f) * 2f * radius;
        float y = (Mathf.PerlinNoise(t + noiseOffsets.y, 1f) - 0.5f) * 2f * radius;
        float z = (Mathf.PerlinNoise(t + noiseOffsets.z, 2f) - 0.5f) * 2f * radius;
        transform.position = basePosition + new Vector3(x, y, z);
    }

    private IEnumerator RunSceneForDuration()
    {
        yield return new WaitForSeconds(duration);
        sslogger.SaveToCSV();
        SceneManager.LoadScene("EyeTrackerreal");
    }
}
