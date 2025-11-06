using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;

public class BallSpawner : MonoBehaviour
{
    [Header("Sphere Settings")]
    [SerializeField] private GameObject saccadeSphere;
    [SerializeField] private float sphereScale = 0.5f;

    [Header("Timing")]
    [SerializeField] private float totalDuration = 30f;
    [SerializeField] private float switchInterval = 0.05f;

    [Header("Horizontal Movement")]
    [SerializeField] private float horizontalDistance = 5f;
    [SerializeField] private float horizontalJitter = 0.5f;

    [Header("Vertical Movement")]
    [SerializeField] private float verticalDistance = 5f;
    [SerializeField] private float verticalJitter = 0.5f;

    [Header("Z Movement")]
    [SerializeField] private float zAmplitude = 2f;
    [SerializeField] private float zSpeed = 0.5f;

    private float elapsedTime;
    private bool isLeft = true;
    private bool isUp = true;
    [SerializeField] private SaccadeLogger slogger;

    private Transform camTransform;
    private Vector3 basePos;
    private const float distanceFromCamera = 4f;

    private void Start()
    {
        if (saccadeSphere == null)
        {
            Debug.LogError("Sphere not assigned!");
            return;
        }

        saccadeSphere.transform.localScale = Vector3.one * sphereScale;
        camTransform = Camera.main != null ? Camera.main.transform : null;
        if (camTransform == null)
        {
            Debug.LogError("Main Camera not found!");
            return;
        }
        basePos = camTransform.position + camTransform.forward * distanceFromCamera;
        StartCoroutine(SaccadePattern());
    }

    private IEnumerator SaccadePattern()
    {
        elapsedTime = 0f;

        while (elapsedTime < totalDuration)
        {
            MoveSphere();
            yield return new WaitForSeconds(switchInterval);
            elapsedTime += switchInterval;
        }
        if (slogger != null)
            slogger.SaveToCSV();
        saccadeSphere.SetActive(false);
        SceneManager.LoadScene("EyeTrackerreal");
    }

    private void MoveSphere()
    {
        float t = Time.time;

        // Calculate saccade offsets
        float z = Mathf.Sin(t * zSpeed) * zAmplitude;
        float x = (isLeft ? -horizontalDistance : horizontalDistance) + Random.Range(-horizontalJitter, horizontalJitter);
        float y = (isUp ? verticalDistance : -verticalDistance) + Random.Range(-verticalJitter, verticalJitter);

        // Offset from base position in camera's local space
        Vector3 offset = camTransform.right * x + camTransform.up * y + camTransform.forward * z;
        saccadeSphere.transform.position = basePos + offset;

        isLeft = !isLeft;
        isUp = !isUp;

        // Debug.Log("Sphere saccade jump to: " + saccadeSphere.transform.position);
    }
}
