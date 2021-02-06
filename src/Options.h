#pragma once

#include <thread>
#include <string>
#include <External/AnyOption/anyoption.h>
#include <Utils/Chrono.cuh>

class Options
{

    // Members
    public:
    bool statistics;
    unsigned int queueSize;
    unsigned int timeout;
    unsigned int widthCpu;
    unsigned int widthGpu;
    unsigned int parallelismCpu;
    unsigned int parallelismGpu;
    unsigned int eqProbability;
    unsigned int neqProbability;
    unsigned int randomSeed;
    char const * inputFilename;
    private:
    AnyOption* const anyOption;

    // Functions
    public:
    Options();
    ~Options();
    bool parseOptions(int argc, char* argv[]);
    void printOptions();
};

Options::Options() :
    statistics(false),
    inputFilename(nullptr),
    queueSize(50000),
    timeout(60),
    widthCpu(1000),
    widthGpu(3),
    eqProbability(15),
    neqProbability(15),
    randomSeed(static_cast<unsigned int>(Chrono::now() % 1000)),
    anyOption(new AnyOption())
{
    // Help
    anyOption->addUsage("Usage: ");
    anyOption->addUsage("");
    anyOption->addUsage(" -h --help             Print this help");
    anyOption->addUsage(" -s                    Print search statistics");
    anyOption->addUsage(" -q <size>             Size of branch and bound queue");
    anyOption->addUsage(" -t <seconds>          Timeout");
    anyOption->addUsage(" --wc <size>           Width of MDDs explored on CPU");
    anyOption->addUsage(" --wg <size>           Width of MDDs explored on GPU");
    anyOption->addUsage(" --pc <count>          Number of MDDs explored in parallel on CPU");
    anyOption->addUsage(" --pg <count>          Number of MDDs explored in parallel on GPU");
    anyOption->addUsage(" --eq <percentage>  	Probability to use a value in large neighborhoods search");
    anyOption->addUsage(" --neq <percentage>  	Probability to discard a value in large neighborhoods search");
    anyOption->addUsage(" --rs <integer>  	    Random seed");
    anyOption->addUsage("");

    anyOption->setFlag("help",'h');
    anyOption->setOption('s');
    anyOption->setOption('q');
    anyOption->setOption('t');
    anyOption->setOption("wc");
    anyOption->setOption("wg");
    anyOption->setOption("pc");
    anyOption->setOption("pg");
    anyOption->setOption("eq");
    anyOption->setOption("neq");
    anyOption->setOption("rs");

}

Options::~Options()
{
     free(anyOption);
}

bool Options::parseOptions(int argc, char* argv[])
{
    anyOption->processCommandArgs(argc, argv);

    if (not anyOption->hasOptions())
    {
        anyOption->printUsage();
        return false;
    }

    if (anyOption->getFlag('h') or anyOption->getFlag("help"))
    {
        anyOption->printUsage();
        return false;
    }

    if (anyOption->getFlag('s'))
    {
        statistics = true;
    }

    if (anyOption->getValue('q') != nullptr)
    {
        queueSize = static_cast<unsigned int>(std::stoi(anyOption->getValue('q')));
    }

    if (anyOption->getValue('t') != nullptr)
    {
        timeout = static_cast<unsigned int>(std::stoi(anyOption->getValue('t')));
    }

    if (anyOption->getValue("wc") != nullptr)
    {
        widthCpu = static_cast<unsigned int>(std::stoi(anyOption->getValue("wc")));
    }

    if (anyOption->getValue("wg") != nullptr)
    {
        widthGpu = static_cast<unsigned int>(std::stoi(anyOption->getValue("wg")));
    }

    if (anyOption->getValue("pc") != nullptr)
    {
        parallelismCpu = static_cast<unsigned int>(std::stoi(anyOption->getValue("pc")));
    }
    else
    {
        unsigned int const coresCount = std::thread::hardware_concurrency();
        parallelismCpu = 4 * coresCount;
    }

    if (anyOption->getValue("pg") != nullptr)
    {
        parallelismGpu = static_cast<unsigned int>(std::stoi(anyOption->getValue("pg")));
    }
    else
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        unsigned int const multiProcessorCount = deviceProp.multiProcessorCount;
        unsigned int const maxBlocksPerMultiProcessor = deviceProp.maxBlocksPerMultiProcessor;
        parallelismGpu = 2 * multiProcessorCount * maxBlocksPerMultiProcessor;
    }

    if (anyOption->getValue("eq") != nullptr)
    {
        eqProbability = static_cast<unsigned int>(std::stoi(anyOption->getValue("eq")));
    }

    if (anyOption->getValue("neq") != nullptr)
    {
        neqProbability = static_cast<unsigned int>(std::stoi(anyOption->getValue("neq")));
    }

    if (anyOption->getValue("rs") != nullptr)
    {
        randomSeed = static_cast<unsigned int>(std::stoi(anyOption->getValue("rs")));
    }

    inputFilename = anyOption->getArgv(0);

    return true;
}
void Options::printOptions()
{
    printf("[INFO] Input file: %s\n", inputFilename);
    printf("[INFO] Queue size: %u\n", queueSize);
    printf("[INFO] Timeout: %u\n", timeout);
    printf("[INFO] Width CPU: %u\n", widthCpu);
    printf("[INFO] Width GPU: %u\n", widthGpu);
    printf("[INFO] Parallelism CPU: %u\n", parallelismCpu);
    printf("[INFO] Parallelism GPU: %u\n", parallelismGpu);
    printf("[INFO] Used values: %u%%\n", eqProbability);
    printf("[INFO] Discarded values: %u%%\n", neqProbability);
    printf("[INFO] Random seed: %u\n", randomSeed);
}
