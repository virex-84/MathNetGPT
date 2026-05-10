//https://github.com/virex-84


using System.Globalization;
using System.Text.Json;

namespace Llm.Core;

public class Dataset
{
    public List<string> PretrainingData { get; set; } = new();
    public List<string> ChatTrainingData { get; set; } = new();
}

public enum DatasetType
{
    Json
}

public static class DatasetLoader
{
    public static Dataset LoadDataset(
        string pretrainingDataPath,
        string chatTrainingDataPath,
        DatasetType typeOfData)
    {
        List<string> pretrainingData;
        List<string> chatTrainingData;

        switch (typeOfData)
        {
            case DatasetType.Json:
                pretrainingData = GetDataFromJson(pretrainingDataPath);
                chatTrainingData = GetDataFromJson(chatTrainingDataPath);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(typeOfData), "Unsupported dataset type");
        }

        return new Dataset
        {
            PretrainingData = pretrainingData,
            ChatTrainingData = chatTrainingData,
        };
    }

    /// <summary>
    /// Вычисляет оптимальную максимальную длину последовательности на основе данных.
    /// Анализирует все тексты, токенизирует их и находит максимальное количество токенов.
    /// </summary>
    /// <param name="texts">Список текстов для анализа</param>
    /// <param name="vocab">Словарь для токенизации</param>
    /// <param name="safetyFactor">Коэффициент безопасности (1.0-1.5) для обобщения</param>
    /// <returns>Рекомендуемая максимальная длина последовательности</returns>
    public static int CalculateOptimalMaxSeqLen(IEnumerable<string> texts, Vocab vocab, float safetyFactor = 1.5f)
    {
        int maxTokenCount = 0;
        
        foreach (var text in texts)
        {
            var tokens = Tokenizer.Tokenize(vocab, text);
            if (tokens.Count > maxTokenCount)
            {
                maxTokenCount = tokens.Count;
            }
        }
        
        var optimalLength = (int)Math.Ceiling(maxTokenCount * safetyFactor);
        // Ограничиваем разумными пределами: минимум 64, максимум 8192
        return Math.Clamp(optimalLength, 64, 8192);
    }

    private static List<string> GetDataFromJson(string path)
    {
        var dataJson = File.ReadAllText(path);
        var data = JsonSerializer.Deserialize<List<string>>(dataJson);
        return data ?? new List<string>();
    }
}
