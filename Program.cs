using System.Text;
using ConsoleApp2;
using Grpc.Net.Client;
using Qdrant.Client;
using Qdrant.Client.Grpc;

Console.WriteLine();


using var ollama = new OllamaService();

using var client = new QdrantClient("localhost");
const string collection = "somewhat2";
if (!await client.CollectionExistsAsync(collection))
    await client.CreateCollectionAsync(collection, new VectorParams()
    {
        Distance = Distance.Cosine,
        Size = 768
    });


// Console.WriteLine(await client.GetCollectionInfoAsync("test_collection"));
await ReadFiles();

while(true)
    await QueryFiles();

async Task QueryFiles()
{
    
    Console.Write("Search query: ");
    string query = Console.ReadLine();
    var embeddings = await ollama.GetEmbedding("search_query: " + query);
    var result = await client.SearchAsync(collection, embeddings, limit: 5);
    StringBuilder context = new();
    foreach (var point in result)
    {
        var file = point.Payload["file"].StringValue;
        var text = point.Payload["text"].StringValue;
        Console.WriteLine($"[{file}] {text[..Math.Min(200, text.Length)]}...");
        // context.AppendLine($"Источник: файл {file}:");
        context.AppendLine($"{text}\n---\n");
    }

    Console.ForegroundColor = ConsoleColor.Green;

    Console.WriteLine("LLM:");
    string prompt = $"Контекст:{context}\nВопрос:{query}\nОтвет:";

    Console.WriteLine(await ollama.GetChatCompletionAsync(prompt));
    
    Console.ForegroundColor = ConsoleColor.White;
}

async Task ReadFiles()
{
    int id = 1;
    foreach (var file in Directory.GetFiles(
                 Path.Join(Directory.GetCurrentDirectory(), "/lib"), "*.txt"))
    {
        Console.Write($"File {file}");
        Console.Write("embeddings ... ");
        string content = File.ReadAllText(file);
        var embeddings = await ollama.GetEmbedding("search_document: " + content);
        Console.WriteLine("ok");
        Console.Write("saving to db ... ");
        foreach (var chunk in ChunkText(content))
        {
            var vec = await ollama.GetEmbedding(chunk);
            await client.UpsertAsync(collection, new[] {
                new PointStruct {
                    Id = (ulong)id++,
                    Vectors = vec,
                    Payload = { ["text"] = chunk, ["file"] = file }
                }
            });
            Console.Write('.');
        }
        Console.WriteLine(" ok");
    }
}

static IEnumerable<string> ChunkText(string text, int maxTokens = 100)
{
    var sentences = text.Split(new[] { '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
    var current = new StringBuilder();
    foreach (var sentence in sentences)
    {
        if (current.Length + sentence.Length > maxTokens * 4) // ~4 символа на токен
        {
            yield return current.ToString();
            current.Clear();
        }
        current.Append(sentence).Append(". ");
    }
    if (current.Length > 0)
        yield return current.ToString();
}
