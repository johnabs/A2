using Evolutionary
using CSV
using DataFrames

#Set Filename and Read CSV into DataFrame
filename="knapPI_16_50_1000.csv"

function parta(file)
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

    #Create objective function, which defaults to 0 if it violates the constraint,
    #and the sum is set to negative as this EA uses minimization
    f(x)=((sum(w.*x)<=con) ? -sum(v.*x) : sum(w.*x))

    #Create bitvector to represent the selected items; provide at least 1 zero, so the objective function will actually return a good score
    #starting out; otherwise it just prematurely converges to 0.
    x0=BitVector(vcat(zeros(n-1),1))

    #Defined uniform bit flipping function.
    function uflip(recombinant::T) where {T <: BitVector}
        p = 0.5
        s = length(recombinant)
        check = rand(s).<p
        for i in 1:length(check)
            if check[i]==1
            recombinant[i]=!recombinant[i]
            end
        end
        return recombinant
    end

    #Defined uniform bit flipping function.
    function swap(recombinant::T) where {T <: BitVector}
        s = length(recombinant)
        p1=rand(1:s)
        p2=rand(1:s)
        while(p2==p1)
            p2=rand(1:s)
        end
        t1=copy(recombinant[p1])
        t2=copy(recombinant[p2])
        recombinant[p1]=t2
        recombinant[p2]=t1
        return recombinant
    end


    #Define different combinations of genetic algorithms with different types of mutation, and types of crossover
    #uniform and singlepoint.
    points=reduce(vcat,map(y->map(x->(x,y),0.1:0.1:0.9),0.1:0.1:0.9))
    gafu=map(x->GA(populationSize=1000,selection=roulette,mutation=flip,crossover=uniform,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);
    gafs=map(x->GA(populationSize=1000,selection=roulette,mutation=flip,crossover=singlepoint,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);
    gauu=map(x->GA(populationSize=1000,selection=roulette,mutation=uflip,crossover=uniform,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);
    gaus=map(x->GA(populationSize=1000,selection=roulette,mutation=uflip,crossover=singlepoint,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);

    gaswu=map(x->GA(populationSize=1000,selection=roulette,mutation=swap,crossover=uniform,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);
    gasws=map(x->GA(populationSize=1000,selection=roulette,mutation=swap,crossover=singlepoint,mutationRate=x[1],crossoverRate=x[2],ɛ = 0.1),points);

    #Run the optimization, suppress output.
    rfu = map(x->Evolutionary.optimize(f, x0, x, Evolutionary.Options(iterations=10000)),gafu);
    fus=hcat(points,-minimum.(rfu),Evolutionary.iterations.(rfu),Evolutionary.f_calls.(rfu))

    rfs = map(x->Evolutionary.optimize(f, x0, x, Evolutionary.Options(iterations=10000)),gafs);
    fss=hcat(points,-minimum.(rfs),Evolutionary.iterations.(rfs),Evolutionary.f_calls.(rfs))

    ruu = map(x->Evolutionary.optimize(f, x0, x, Evolutionary.Options(iterations=10000)),gauu);
    uus=hcat(points,-minimum.(rfs),Evolutionary.iterations.(rfs),Evolutionary.f_calls.(rfs))

    rus = map(x->Evolutionary.optimize(f, x0, x, Evolutionary.Options(iterations=10000)),gaus);
    uss=hcat(points,-minimum.(rfs),Evolutionary.iterations.(rfs),Evolutionary.f_calls.(rfs))

    rswu = map(x->Evolutionary.optimize(f, x0, x, Evolutionary.Options(iterations=10000)),gaswu);
    swus=hcat(points,-minimum.(rfs),Evolutionary.iterations.(rfs),Evolutionary.f_calls.(rfs))

    rsws = map(x->Evolutionary.optimize(f, x0, x, Evolutionary.Options(iterations=10000)),gasws);
    swss=hcat(points,-minimum.(rfs),Evolutionary.iterations.(rfs),Evolutionary.f_calls.(rfs))
end

parta(filename)

#Starting for part B
#f(rand(Bool,50))
