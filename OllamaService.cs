using System.Text;
using System.Text.Json;

namespace ConsoleApp2;

public class OllamaService : IDisposable
{
    private readonly HttpClient _client;
    private const string EmbeddingsModel = "nomic-embed-text";
    private const string ChatModel = "qwen3:8b";

    public OllamaService()
    {
        _client = new HttpClient { BaseAddress = new Uri("http://localhost:11434") };
    }

    public async Task<float[]> GetEmbedding(string text)
    {
        var request = new
        {
            model = EmbeddingsModel,
            prompt = text
        };

        var json = JsonSerializer.Serialize(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        var response = await _client.PostAsync("/api/embeddings", content);
        response.EnsureSuccessStatusCode();

        var jsonResponse = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(jsonResponse);
        var embeddings = doc.RootElement.GetProperty("embedding");
        using var array = embeddings.EnumerateArray();
        return array.Select(x=>x.GetSingle()).ToArray();
    }

    public async Task<string> GetChatCompletionAsync(string prompt)
    {
        var request = new
        {
            model = ChatModel,
            prompt = prompt,
            stream = false
        };

        var content = new StringContent(JsonSerializer.Serialize(request), Encoding.UTF8, "application/json");
        var response = await _client.PostAsync("/api/generate", content);
        response.EnsureSuccessStatusCode();

        var jsonResponse = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(jsonResponse);
        var responseText = doc.RootElement.GetProperty("response").GetString();

        return responseText ?? string.Empty;
    }
    
    public void Dispose()
    {
        _client.Dispose();
    }
}
