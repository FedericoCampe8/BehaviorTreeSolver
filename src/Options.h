#pragma once

#include <string>
#include <External/AnyOption/anyoption.h>
#include <Utils/Chrono.cuh>
#include <Utils/TypeAlias.h>

class Options
{

    // Members
    public:
    bool statistics;
    u32 queueSize;
    u32 timeout;
    u32 widthCpu;
    u32 widthGpu;
    u32 parallelismCpu;
    u32 parallelismGpu;
    float eqProbability;
    float neqProbability;
    u32 randomSeed;
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
    queueSize(0),
    timeout(0),
    widthCpu(0),
    widthGpu(0),
    eqProbability(0),
    neqProbability(0),
    randomSeed(static_cast<u32>(Chrono::now() % 1000)),
    anyOption(new AnyOption())
{
    // Help
    anyOption->addUsage("Usage: ");
    anyOption->addUsage("");
    anyOption->addUsage(" -h --help             Print this help");
    anyOption->addUsage(" -s                    Print search statistics");
    anyOption->addUsage(" -q <size>             Size of the queue for the initial search");
    anyOption->addUsage(" -t <seconds>          Timeout");
    anyOption->addUsage(" --wc <size>           Width of MDDs explored on CPU");
    anyOption->addUsage(" --wg <size>           Width of MDDs explored on GPU");
    anyOption->addUsage(" --pc <count>          Number of MDDs explored on CPU");
    anyOption->addUsage(" --pg <count>          Number of MDDs explored on GPU");
    anyOption->addUsage(" --eq <percentage>  	Probability to use a value in a neighborhood during LNS");
    anyOption->addUsage(" --neq <percentage>  	Probability to discard a value in a neighborhood during LNS");
    anyOption->addUsage(" --rs <integer>  	    Random seed");
    anyOption->addUsage("");

    anyOption->setFlag("help",'h');
    anyOption->setFlag('s');
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
        queueSize = static_cast<u32>(std::stoi(anyOption->getValue('q')));
    }

    if (anyOption->getValue('t') != nullptr)
    {
        timeout = static_cast<u32>(std::stoi(anyOption->getValue('t')));
    }

    if (anyOption->getValue("wc") != nullptr)
    {
        widthCpu = static_cast<u32>(std::stoi(anyOption->getValue("wc")));
    }

    if (anyOption->getValue("wg") != nullptr)
    {
        widthGpu = static_cast<u32>(std::stoi(anyOption->getValue("wg")));
    }

    if (anyOption->getValue("pc") != nullptr)
    {
        parallelismCpu = static_cast<u32>(std::stoi(anyOption->getValue("pc")));
    }

    if (anyOption->getValue("pg") != nullptr)
    {
        parallelismGpu = static_cast<u32>(std::stoi(anyOption->getValue("pg")));
    }

    if (anyOption->getValue("eq") != nullptr)
    {
        eqProbability = static_cast<float>(std::stof(anyOption->getValue("eq")));
        assert(eqProbability <= 1.0);
    }

    if (anyOption->getValue("neq") != nullptr)
    {
        neqProbability = static_cast<float>(std::stof(anyOption->getValue("neq")));
        assert(neqProbability <= 1.0);
    }

    if (anyOption->getValue("rs") != nullptr)
    {
        randomSeed = static_cast<u32>(std::stoi(anyOption->getValue("rs")));
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
    printf("[INFO] Probability of using a value: %.3f\n", eqProbability);
    printf("[INFO] Probability of discarding a value: %.3f\n", neqProbability);
    printf("[INFO] Random seed: %u\n", randomSeed);
}
