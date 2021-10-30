using Evolutionary
using CSV
using DataFrames

function getInstance(file)
    data=CSV.read(file,DataFrame,skipto=6,header=false)

    #Also read in constraint value
    con=open(file) do file
        fileText= readlines(file)
        con=fileText[3][3:end]
        return parse(Int,con)
    end

    #Create variables, knapsack options (n), weights (w), and values (v)
    n=size(data)[1]
    w=data[:,3]
    v=data[:,2]
    return con,n,w,v
end

function bestPartA(con,n,w,v,popSize)
    
    mutRate = 0.9 #UPDATE LATER
    crossRate = 0.9 #UPDATE LATER
    
    #Create objective function, which defaults to 0 if it violates the constraint,
    #and the sum is set to negative as this EA uses minimization
    f(x)=((sum(w.*x)<=con) ? -sum(v.*x) : sum(w.*x))

    #Create bitvector to represent the selected items; provide at least 1 zero, so the objective function will actually return a good score
    #starting out; otherwise it just prematurely converges to 0.
    x0 = BitVector(vcat(zeros(n-1),1)) 

    ga = GA(populationSize=popSize,selection=roulette,mutation=flip,crossover=uniform,mutationRate=mutRate,crossoverRate=crossRate,ɛ = 0.1);
    gaResults = Evolutionary.optimize(f, x0, ga, Evolutionary.Options(iterations=10000))
    
    return gaResults #technically unecessary, but helps me think
end

function partb(file)
    con,n,w,v = getInstance(file)
    popSize = 1000 #might not need to vary population size? TBD
    gaResults = bestPartA(con,n,w,v,popSize)
    f(x) = -sum(v.*x)
    
    function randSampling()
        function repair(x)
            while sum(w.*x) > con
                x[rand(findall(x))] = false
            end
            return x
        end

        numSolns = Evolutionary.f_calls(gaResults)
        solutions = repair.([rand(Bool,n) for _ in 1:numSolns])
        bestFitness = minimum(f.(solutions))
        return bestFitness
    end

    randBestFitness = -randSampling()
    gaBestFitness = -Evolutionary.minimum(gaResults)
    return gaBestFitness, randBestFitness
end

#Set Filename and Read CSV into DataFrame
filename="knapPI_16_50_1000.csv"
partb(filename)