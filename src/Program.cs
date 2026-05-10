//https://github.com/virex-84

using Llm.Core;
using Newtonsoft.Json;

public class Program
{
    private const string ModelPath = "llm_model.bin";

    public static void Main(string[] args)
    {
        LLM llm;

        var pretrain = JsonConvert.DeserializeObject<List<string>>(File.ReadAllText("data\\pretrain.json"));
        var tune = JsonConvert.DeserializeObject<List<string>>(File.ReadAllText("data\\tune.json"));

        if (File.Exists(ModelPath))
        {
            Console.WriteLine("\n=== LOADING MODEL ===");
            llm = LLM.LoadModel(ModelPath);
            Console.WriteLine("Model loaded from disk.");
        }
        else
        {
            Console.WriteLine("\n=== TRAINING NEW MODEL ===");


            var vocabSet = new HashSet<string>();

            Vocab.ProcessTextForVocab(pretrain, vocabSet);
            Vocab.ProcessTextForVocab(tune, vocabSet);

            var vocabWords = vocabSet.ToList();
            vocabWords.Sort();
            var vocab = new Vocab(vocabWords);

            // Вычисляем оптимальную максимальную длину последовательности на основе данных
            var allTexts = pretrain.Concat(tune);
            var maxSeqLen = DatasetLoader.CalculateOptimalMaxSeqLen(allTexts, vocab);
            Console.WriteLine($"Вычислена оптимальная MAX_SEQ_LEN: {maxSeqLen}");

            var embeddingDim = 128;
            var hiddenDim = 256;

            var transformerBlock1 = new TransformerBlock(embeddingDim, hiddenDim);
            var transformerBlock2 = new TransformerBlock(embeddingDim, hiddenDim);
            var transformerBlock3 = new TransformerBlock(embeddingDim, hiddenDim);
            var outputProjection = new OutputProjection(embeddingDim, vocab.Words.Count);
            var embeddings = new Embeddings(vocab, maxSeqLen);
            llm = new LLM(vocab, maxSeqLen, new List<ILayer>
            {
                embeddings,
                transformerBlock1,
                transformerBlock2,
                transformerBlock3,
                outputProjection
            });

            Console.WriteLine("\n=== PRE-TRAINING MODEL ===");
            llm.Train(pretrain, 10, 0.0005f);

            var result = llm.PredictWithConfidence("Звезды");
            Console.WriteLine($"Звезды...\nModel output: {result.Text}");

            Console.WriteLine("\n=== INSTRUCTION TUNING ===");
            llm.Train(tune, 10, 0.0001f);

            var result2 = llm.PredictWithConfidence("User: Что такое электричество?");
            Console.WriteLine($"User: Что такое электричество?...\nModel output: {result2.Text}");

            Console.WriteLine("\n=== SAVING MODEL ===");
            llm.Save(ModelPath);
            Console.WriteLine("Model saved to disk.");
        }

        Console.WriteLine("\n=== MODEL INFORMATION ===");
        Console.WriteLine($"Network architecture: {llm.NetworkDescription()}");
        Console.WriteLine($"Total parameters: {llm.TotalParameters()}");

        Console.WriteLine("\n--- Interactive Mode ---");
        Console.WriteLine("Type a prompt and press Enter to generate text.");
        Console.WriteLine("Type 'exit' to quit.");
        Console.WriteLine("Type 'tune' to fine-tune 10 epochs");
        Console.WriteLine("Type 'save' to save model.");

        while (true)
        {
            Console.Write("\nEnter prompt: ");
            var input = Console.ReadLine();

            if (string.IsNullOrEmpty(input))
                continue;

            var command = input.Trim().ToLowerInvariant();

            if (command.Equals("exit", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("Exiting interactive mode.");
                break;
            }

            if (command.Equals("tune", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("tune 10 epoch...");
                llm.Train(tune, 10, 0.0001f);
                continue;
            }

            if (command.Equals("save", StringComparison.OrdinalIgnoreCase))
            {
                llm.Save(ModelPath);
                Console.WriteLine("Model saved to disk.");
                continue;
            }

            // Обычный запрос к модели
            var formattedInput = $"User: {input.Trim()}";
            var result = llm.PredictWithConfidence(formattedInput);

            //if (result.MinConfidence > 0.6f)
                Console.WriteLine($"\nModel output: {result.Text}");
            //else
                //Console.WriteLine("Model output: Затрудняюсь ответить");
        }
    }
}