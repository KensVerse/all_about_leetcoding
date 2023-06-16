// g++ -o main.exe ./main.cpp ./increment_and_sum.cpp ./vect_add_one.cpp
// ./a.exe ./a.out for linux
/*
comment
*/
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <cstring>
#include <cmath>
#include <cassert>
#include <stdio.h> // printf("%s \n", "A string");
#include <stdlib.h> // malloc
#include <memory> // smartpointer
#include <thread>
#include <future>
#include <chrono>
#include <mutex>
#include <queue>
#include <random>
#include <set>
#include <unordered_set>
using std::cout;
using std::vector;
using std::ifstream;
using std::istringstream;
using std::string;
using std::unordered_map;
using std::sort;
using std::abs;
using std::find;
using std::unordered_multiset;
#define PI 3.14159;  // define macro

int main() {
    std::vector<std::string> brothers{"David", "Ethan", "Adam"};
    for (std::string const& brother: brothers){
        std::cout << "Hello " << brother << "!\n";
    }
}
for (int l=0, r=num.size()-1; l <= r; l++, r--){} // size() returns a long int, which can only be added to an int after casting using int()
for (int i = 0, m = words.size(); i < m; i++)
for (int j=i+1; s.count(j); j++) s.erase(j), cnt++;


count += 3 & 7 > 0;// 3 & 7 == 3
printf("a&b = %d\n", a & b);
i << 2; // i multiplied with 2
popcount(3) // count bit of 1 in binary form
int('0') == 48, use int('0') - '0'
for (i = 1000; i >= 0 && a[i] != 1; --i); // one line suffices, stops when a[i] == 1
for(int i = 0, sum = 0; i < 1001;)
stoi(color.substr(i,2), nullptr, 16); // convert string into integer 16-bit
long int x;
int x = INT_MAX;// INT_MIN
DBL_MAX
max(integer1, integer2);
count_if(as.begin(), as.end(), [&](auto a){return exist.count(a + 1);
% // modulo, size() need to be transformed using int()
res += T > upper ? 1 : T < lower ? -1 : 0;
groups += !visited[i] ? dfs(i, M, visited), 1 : 0; // if true, run dfs and add 1, otherwise add 0
int i = 0, j = 0;
int n = A.size(), d = (A[n - 1] - A[0]) / n
std::log10()
if (root->left) //nullptr as boolean
word[i++] != abbr[j++] // indexing using i and j, then increment by 1 each
rand()
isdigit(IAmAChar), islower(IAmAChar), isupper(achar)
float(an_integer)
char foo [20]; // char array, containing 20 characters
abs(x1 - x2)
!open.empty()
&&， || // if condition "and" and "or"
#define printVarValue(x) cout<<"name => **"<<(#x)<<"** value => "<<x<<"\n"
std::to_string(Day()) // convert to string
astring.substr(0, idx + 1)
std::copy ( myints, myints+7, myvector.begin() ); // deep copy
std::ostringstream stream;
stream << "age" << ": " << 20;
return stream.str(); // age: 20
// calling function B within function A, A's local variables are unknown to B
{} // is a scope, everything defined inside will be destructed when leaving the bracket


if a is a pointer of int, a[0] = 1 would work
// array
int myArray[10] = {0} initialize all elements to 0
int s[2]; // array variable itself is a pointer
int dict[256];
fill_n(dict,256,0); // fill the first 256 places with 0
int *p = s;
p[0] = 1;
p[1] = 2;
if a is a pointer of array's start, a + 1, a + 2... and *(a + 1), *(a + 2)... would work
// concatenate array
int * result = new int[size1 + size2];
std::copy(arr1, arr1 + size1, result);
std::copy(arr2, arr2 + size2, result + size1);

// vector
vector<int> v; // vector has to be first dereferenced and then use index
vector<vector<int>> vec( n , vector<int> (m, 0)); //vector<vector<int>> vec( n , vector<int>());
vector<int>(a.begin() + 1, a.end() - 1);
{it, it + k} // it... pointer, returns a vector
equal(good.begin() + i + 1, good.end(), bad.begin() + i, bad.end() - 1);
vector<int> good(n,1) // initialize a vector of size n with elements of 1
vector1.insert( vector1.end(), vector2.begin(), vector2.end() ) // concatenate 2 vectors
reverse(vec1.begin(), vec1.end())
reverse(vec1.begin(), vec1.begin() + 2) // reverse only the first 2 elements
swap(arr1[i], arr[j])
// deep copy
vect2.assign(vect1.begin(), vect1.end());
vect2.insert(vect2.begin(), vect1.begin(), vect1.end());
copy(vect1.begin(), vect1.end(), back_inserter(vect2));
v.push_back(906);
v.back(): the last item
pop_back() ; remove the last item
upper_bound(begin(n), end(n), t) // return the pointer of the first pointer to the element which is larger than target (t)
const auto& [f, s] = equal_range(nums.begin(), nums.end(), target);// pair of iterators, start and end idx equal to target
distance(f, s)
vector<int> * p = &v;
cout << (*p)[0] << endl;
accumulate(begin(s), end(s), 0.)
reduce(arr.begin(), arr.end())
*min_element(begin(s), end(s)) // returns the mininum value; max_element
find(months_30.begin(), months_30.end(), month) != months_30.end() // vector
curset.erase(std::remove(curset.begin(), curset.end(), key), curset.end()); // remove all occurences of key
vec.erase(std::next(vec.begin(), 1), std::next(vec.begin(), 3));
nodecol.erase(nodecol.begin()); // erase the first item
v.insert(v.begin(), 6); // insert a value at the beginning

//unique elemeent
set<int> s(vec.begin(), vec.end());
vec.assign( s.begin(), s.end() ); // if a vector is needed
//alternatively
sort(vec.begin(), vec.end());
sort(begin(vec)., end(vec));
vec.erase( unique( vec.begin(), vec.end() ), vec.end() );

std::deque also uses .back(), pop_back(), .empty(), push_back()
std::queue uses .front()(first added item), .pop()(remove first added item), .push(add new item)

// set
unordered_set<string> s;
s.erase(7);
set<int> s(vec.begin(), vec.end());
set<int> s;
s.insert(vec[i])
s.count(x)
container.find(element) != container.end() // before c++ 20
container.contains(element) //after c++20
unordered_multiset<int> nums {1, 3, 4, 1};
nums.count(1); // return 2, can contain duplicate values

bitset<256> hash
hash.flip(character)
hash.count

// dict
std::map<std::string, int> map
map["a"] += 1; // initialized with 0
map.erase("a")
unordered_map <int, string> my_dictionary {{5, "a"}};
counter["cat"]++;
for (auto pair : counter) cout << pair.first << ":" << pair.second << std::endl;
my_dictionary.count(key) // if 1, key exists in my_dictionary
my_dictionary.find(key) == my_dictionary.end() // find out whether "key" exists in the dict (true if non existence)
my_dictionary[key] // returns the value associated with the key
for (auto it = mm.begin(); it != mm.end(); ++it) if(it->second == 1)

// char
char: 's', string: "s" // char single quote, while string double quotes
string and char can be concatenated using +
char c1 = '0';
std::string(1,c1); // convert char into string
astring[0] is a char type
char *brand; // character array as function parameter, can set a string equal to char array
char brand[] = "Peugeot"; // can be thrown in as function argument
Car::brand = new char[brand_name.length() + 1];// given a string, set to char array
strcpy(Car::brand, brand_name.c_str());
result += Car::brand; // add char array to a string to become an longer string
'a' can be put as index in an array as index 97

// string
std::string b = "Here is a string";
.empty()
rotate(vec1.begin(), vec1.begin()+rotL, vec1.end()); // shifting
b.resize(5) // b retains only the first 5 letters
b.substr(3) // from 4th element until the end
b.substr(3, 5) // from 4th element to 8th element
string str = "a123";
int count;
int found = text.find(w, pos) // find a substring in a string, starting from "pos", pos default to 0
found == string::npos // substring not found
keyboard.find_first_of(astring) // find the first occurance of any char within the argument string
S.erase(remove_if(S.begin(), S.end(), [&](char c){ return vowels.find(c) != vowels.end();}), S.end());

// stream
istringstream my_stream(str);
my_stream >> c >> count; // a and 123
// read string
string b("1,2,3p");
istringstream my_stream(b);
char c;
int n;
while (my_stream >> n >> c) {
  cout << "That stream was successful:" << n << " " << c << "\n";
}
// hex format
stringstream st;
st << hex << uppercase  << n;
string s(st.str());
// convert hex to decimal
string hexStr = "0x3ae";
unsigned int x;
stringstream ss;
ss << std::hex << hexStr;
ss >> x;


throw std::invalid_argument("negative dimension"); // followed by if statement, error catcher
vector<vector<int>> v {{1,2}, {3,4}};
auto v_6 = {1, 2, 3};
v_6.size()
for (int i=0; i < 5; i++)
if (a) {}
else if (){}
std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
// read file
std::ifstream my_file;
my_file.open("files/1.board"); // or just std::ifstream my_file(path)
std::string line;
while (getline(my_file, line)) {
    std::cout << line << "\n";
}
// try catch block
bool caught{false};
try {
  Pyramid fyramid(-1, 2, 3);
} catch (...) {
  caught = true;
}





std::cin >> i; // input a value and assign to i
const int j = i * 2;  // "j can only be evaluated at run time.", use const most of the time
                      // "But I promise not to change it after it is initialized."
constexpr int k = 3;  // "k, in contrast, evaluated at compile time, the right side must be a numeric value, not a variable name
int sum(const std::vector<int> &v){} // passed in by reference, but cannot change it; if not passed by reference, create a copy of it in the memory residing on another address
int MultiplyByTwo(int &i) {} //modify the input parameter
int& j = i; // whenever i/j changes, j/i changes as well; if & appears on the RHS, it denotes a pointer; As a decent rule of thumb, references should be used in place of pointers when possible
for (auto& i: v) { // by reference
    i++; // for each element in v, increment each element by 1
}
// sort
sort(intervals.begin(), intervals.end(), [](Interval& l, Interval& r){return l.start < r.start;});
void CellSort(vector<vector<int>> *v) { // descending, input is a pointer, pointer is much more efficient than a copy of a huge vector as function parameter
  sort(v->begin(), v->end(), Compare); // compare: custom boolean function determining comparison between 2 values
}
CellSort(&open); // open is a 2-dimensional vector of integers, pointer of open is the input; *"pointer var" returns the original value
vector<int> currentNode = open.back(); // last item
open.pop_back(); // pop last item
// won't work, as the pointer is not allocated a space
int *px;
*px = 10;

// recursive
int closestValue(TreeNode* root, double target) {
    int a = root->val;
    auto kid = target < a ? root->left : root->right;
    if (!kid) return a;
    int b = closestValue(kid, target);
    return abs(a - target) < abs(b - target) ? a : b;
}

void dfs(int i, vector<vector<int>>& M, vector<bool>& visited) {
    visited[i] = true;
    for (int j = 0; j < visited.size(); j++) {
        if (i != j && M[i][j] && !visited[j]) {
            dfs(j, M, visited);
        }
    }
}

// listnode
ListNode* deleteNodes(ListNode* head, int m, int n) {
    ListNode temp(0, head);
    for (auto p = &temp; p != nullptr; )// temp is used as reference, not as a copy
        for (int i = 0; i < n + m && p != nullptr; ++i)
            if (i < m)
                p = p->next;
            else if (p->next != nullptr)
                p->next = p->next->next;
    return head;
}

const int delta[4][2]{{-1, 0}, {0, -1}, {1, 0}, {0, 1}}; //  can be used as global variable, but it's fixed lengthed
vector<vector<State>> Search(vector<vector<State>> grid, int init[2], int goal[2])
enum class Color {white, black, blue, red}; // content can be in headerfile, no quotation mark need around white or black
Color my_color = Color::blue;
switch(state){
    case State::kObstacle: return "⛰️   ";
    default: return "0   ";
}


class Alien { // one can instantiate an object Alien within definition of Alien
  public: // if private, variables cannot be changed outside th class
    void PrintCarData() {}
    void IncrementDistance() {this->distance++} // this pointer
    Alien(string c) : color(c){}; // constructor. if attribute is a reference or const, it must be initialized using initializer list, not within the constructor
    ~Alien(){}; // Destructor
    // also if the argument is a custom class and if that class constructor needs an argument, then use initializer list instead of constructor body as the class cannot be initialized without a parameter

    // when using initializer list, it can be also a function of members, e.g. 3 * pi
    Alien(string c) {color(c);} // alternative constructor, if function color exists and it has qualified invariant
    string const color; // cannot be changed after initialization
    static float Volume(int radius){} // static function, can be called even when no class object instantiated
  private:
    static float constexpr pi{3.14}；// use constexpr if assigning value within the class definition
    static float const pi；// use const if not assign values; static value is always the same for all instances of the class objects
    static float pi; // if not const or constexpr, then must be defined oustide the class (global) or defined again in the class's cpp file
    friend class Human; // Human class can assess private members of Alien's class
};
float const Alien::pi{3.14}; //abstraction, define Alien's pi
float Alien::pi{3.14};
alien1.color

class Jupiter_Alien : public Alien {}; // has access to all public methods and attributes
class Neptune_Alien: private Alien{}; // within Neptune, Alien's functions can be used
// but when neptune object is instantiated, Alien's functions cannot be applied, as all public
// and protected members of Alien's class are "private" members of Neptune
class Creature: public Alien, public Human {}; // inherit from both human and alien
// but if Alien and Human coming from the same abstract class and inherit the same method, it will fail, called "diamond problem"
Car(): wheels(4, Wheel()){} //composition, constructor using another class Wheel and vector
std::vector<Wheel> wheels;
// Polymorphism: overloading (same function name depending on input parameter type)
int& operator()(int row, int column){values_[row*columns+col];} //operator overloading (), reference used to change value directly there
virtual void Talk() const = 0; // pure Virtual (abstract): this class cannot be instantiated, but all its derived classes must have "talk" defined;
// without "=0" would mean that it is a virtual function, can be instantiated, and derived class must define this function
double Area() const override { return pow(radius_, 2) * PI; } // derived class inplement the function using "override"

void Car::PrintCarData() // define methods outside the class
carpointer->PrintCarData(); // dereference and apply method
vector<Car*> car_vect; // vector of pointers of cars
nullptr // null pointer
cp = new Car(colors[i%3], i+1); // create car pointer in a loop and memory on the heap

struct Date { // access modifier (private, public, protected) also apply to classes; struct preferred over class if attributes independent
  public:// all struct members default to public, while all class members default to private
    void Day(int d){day = d;} // could add condition "invariant" to limit the value, mutator
    int Day() const {return day;} // accessor function, const (attribute not to be modified)
  private:// access specifier; if "protected", atrributes can be assessed inside derived classes
    int day{31}; // cannot access attribute directly, function can be private too
};
// encapsulation: group related data together in a class, protect em from being modified
// abstraction: hide the details from the users, only provide interface (e.g. build filepaths based on top of strings; declare a function and define it later)
void Date::Day(int d){day = d;} // class function can be defined later (scope resolution)
Date day_1(10); //initialize using the constructor
namespace English {void Hello() {;}} // namespace
English::Hello();

template <typename Type> // template
Type Sum(Type a, Type b) {
  return a + b;
}
std::cout << Sum<double>(20.0, 13.7) << "\n";
std::cout << Sum(20.0, 13.7) << "\n";// one can omit the type => template deduction
std::vector v{1, 2, 3}; // template deduction omitting type, g++ -std=c++17 is needed
template <class T>
class MyClass
{T *data = nullptr;}
MyClass<double> h1;// to initiate instance, every time u use "MyClass", u must add "MyClass<double>" or "MyClass<int>"
// class mapping
template <typename KeyType, typename ValueType>
class Mapping {
public:
  Mapping(KeyType key, ValueType value) :key(key), value(value){}
};
Mapping<std::string, int> mapping("age", 20);

// debuger: gdb, only works with main.exe created by "g++ -g" or using visual studio's default "run/debug c/c++ file"
gdb ./main // executable
list // shows the script
break 5 // set breakpoint at line 5, assume it is a string "udacity"
then type"run" and "step" (or "next") and then "p str1" (print the variable str1) or "p &str1" (print the address of the variable)
x/7xb address(7: the subsequent 7 items, t: binary format, b: byte) : print the 7 items in binary byte format after the input argument "address"
memory address always are consecutive numbers
// memory: CPU => Cache => Ram => fixed storage
int x[4][4] // numbers in the first rows are located next to each other, when writing double "for loop", access nearby elements one after each other
// afterwards access the next row, so index order matters
heap: variable will not be deleted when the scope is left, one can access it
when the address is returned; memory allocated at runtime, tailored to actual length

# heap memory reserve // use Valgrind (Memcheck) to detect undelocated memory leaks, after installation type in terminal:
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=./bin/valgrind-out.txt ./bin/memory_leaks_debugging, control F for "leak summary" and "definitely lost", copy the "#size byte" and look for it
for windows: Visual Studio debugger and C Run-time Library (CRT) (bookmarked)
#include stdlib.h // both malloc and calloc return a pointer of type void, but class constructor are not called
// pointer_name = (cast-type*) malloc(size);
int *p = (int*)malloc(3*sizeof(int));
p[0] = 1; p[1] = 2; p[2] = 3;
p + 0, p + 1... *(p + 0), *(p + 1)... would work too
p = (int*)realloc(p,4*sizeof(int));
free(p) // release
// pointer_name = (cast-type*) calloc(num_elems, size_elem);
MyStruct *p = (MyStruct*)calloc(4,sizeof(MyStruct));

// new constructor
vector<double>* r_vec = new vector<double>();
MyClass *myClass = new MyClass();
myClass->setNumber(42); // works as expected
delete myClass;
// heap array
int *p = new int[3];
delete[] p;
// heap vector
vector<int> *random_numbers = RandomNumbers1();
delete[] random_numbers;

void *memory = malloc(sizeof(MyClass));
MyClass *object = new (memory) MyClass;
object->~MyClass();
free(memory);

class MyClass
{
    void *operator new(size_t size) // overload new operator; void *operator new[](size_t size), an array of MyClasses
    {
        void *p = malloc(size);
        return p;
    }
    void operator delete(void *p){free(p);} // void operator delete[](void *p)
};
MyClass *p = new MyClass();//MyClass *p = new MyClass[3]();
delete p;//delete[] p;

// copying policy; rule of 3: if one of those (copy constructor, copy assignment operator, destructor) is overloaded, the other 2 need to be overloaded
DeepCopy(int val){//private member: int *_myInt
    _myInt = (int *)malloc(sizeof(int));
    *_myInt = val;}
~DeepCopy(){
    free(_myInt);}
// 1. no copying policy: "=" and shallow copy constructor won't work
private:
  NoCopyClass1(const NoCopyClass1 &);
  NoCopyClass1 &operator=(const NoCopyClass1 &);
or
  NoCopyClass2(const NoCopyClass2 &) = delete;
  NoCopyClass2 &operator=(const NoCopyClass2 &) = delete;
// 2. deepcopy
DeepCopy(DeepCopy &source){
    _myInt = (int *)malloc(sizeof(int));
    *_myInt = *source._myInt;}
DeepCopy &operator=(DeepCopy &source){
    _myInt = (int *)malloc(sizeof(int));
    *_myInt = *source._myInt;
    return *this;}

MyClass obj3 = obj1; // copy constructor, as it instantiates a new class object
obj2 = obj1 // copy assignment operator
MyClass myClass2(myClass1); // shallow copy constructor, different memory address for class objects, but class attributes share the same address
int i = 1;
int j = i; //deepcopy of integer; int &j = i: shallow copy, shares the same address, lvalue reference
int &&l = i + j; // rvalue reference, where the address of temporary object (i + j) is held, better than creating rvalue i+j then copying and deleting it

int i = 23;
myFunction(std::move(i)); // convert lvalue to rvalue reference, myFunction takes rvalue reference as function argument

MyMovableClass(MyMovableClass &&source){// move constructor, source will be invalidated, transfer source's data's handle to a new object
  _data = source._data;
  _size = source._size;
  source._data = nullptr;
  source._size = 0;}
MyMovableClass &operator=(MyMovableClass &&source){ // move assignment operator
  delete[] _data;// destructor
  ..........}
rule of 5: destructor, assignment operator, copy constructor (can be shallow copying elements), move constructor, move assignment operator
ChatBot & ChatBot::operator=(ChatBot &&source){} // move assignment operator
MyMovableClass obj1(100), obj2(200); // constructor
MyMovableClass obj3(obj1); // copy constructor
obj1 = MyMovableClass(200); // move assignment operator
MyMovableClass obj2 = MyMovableClass(300); // move constructor
MyClass<double> h2 = move(h1); // template
vector<int> random_numbers_3 = RandomNumbers3(); // using move semantics under the hood


void useObject(MyMovableClass obj) would make a temp copy of obj => expensive, better to make it rvalue
void useObject(std::move(obj1)) is better, because no copy constructor is called for the temp copy, local argument has accepted transfer from obj1 (which became invalidated)
_msgqueue.send(std::move(_currentPhase)) // _currentPhase is an element of the current class; still within the class even tho it got moved
// resource acquisition is initialization
class MyInt
{
    int *_p; // pointer to heap data
public:
    MyInt(int *p = NULL) { _p = p; }
    ~MyInt(){delete _p;}
    int &operator*() { return *_p; } // overload dereferencing operator
};
MyInt en(new int(i));

// smart pointer
std::unique_ptr
std::shared_ptr // when internal count reaches 0, deallocated
std::weak_ptr // same as shared_ptr, but no count

std::unique_ptr<int> unique(new int); // create a unique pointer on the stack, heap memory gets released when it goes out of scope, cannot be copied, only can be moved
*unique = 2; // pointer lives on the stack, but its content is on the heap
std::unique_ptr<MyClass> uniquePtr = std::make_unique<MyClass>(23); // rhs often used
unique.get() //get the stack addresses of the pointer
auto destPtr = std::move(sourcePtr) // after std::move it's still a unique pointer
std::move(std::unique_ptr) as unique_ptr is unique, object cannot exist in 2 instances
cannot transform rawptr into smart pointer, can only use smart ptr's .get() as address (aka raw pointer) of the smart pointer


std::shared_ptr<int> shared1(new int); // share ptr persisits until the end of the scope, cannot be "moved" to other types
std::shared_ptr<WaitingVehicles> queue(new WaitingVehicles); // share pointer as the queue is shared between multiple threads
shared1.use_count() // returns how many objects point to the memory on the heap
std::shared_ptr<int> shared2 = shared1;
std::shared_ptr<int> sharedPtr1 = std::move(uniquePtr);
std::shared_ptr<int> sharedPtr2 = weakPtr.lock();
shared->propfunction
shared.reset(new MyClass); // destoys the previous instance and assign a new instance
int *rawPtr = sharedPtr2.get();// move semantics
delete rawPtr; // invalid, as the pointer is alrdy deleed on sharedPtr2

std::weak_ptr<int> myWeakPtr1(mySharedPtr);// weak ptr only created through shared ptr (doesn't increase count) and other weak ptr; breaks cycle of shared_ptr
std::weak_ptr<int> myWeakPtr2(myWeakPtr1);
myWeakPtr.expired() == true // check whether the weak ptr's content got destructed

// function parameters
function should only accept smart pointers as parameters if they access to change the count of shared ptr or transfer ownership from unique_ptr to another
void f(std::unique_ptr<MyClass> ptr){}// is called by:
f(std::move(uniquePtr)) // by using move, uniquePtr is not valid anymore after calling function f
f(unique_ptr&)// pass by reference
unique_ptr not recommended to use with const, as the function only makes sense if ptr is modified

f(sharedPtr) // count will temporily increased inside the f function
f(weakPtr)
f(sharedPtr&) // pass by reference

we pass raw ptrs and references when no changes are needed for them inside the functions, use smart_ptr.get() (raw ptr) as function parameter,
but don't delete them or create new smart ptrs from them

//return type for smart ptr
return a smart ptr only when caller needs to manipulate or access the pointer properties, otherwise use raw pointers
only return by value (due to built-in move semantics)

smart pointer: when raw ptr is needed as parameter or output, use .get(); when object->propfunction is needed, "object" is the smart ptr itself
if a class's member is a unique pointer, it can be initialized in a constructor using a raw pointer
auto newNode = std::find_if(_nodes.begin(), _nodes.end(), [&id](std::unique_ptr<GraphNode> &node) { return node->GetID() == id; });// node passed by reference, as it is assigned to a new node

class Vehicle : public TrafficObject, public std::enable_shared_from_this<Vehicle>{ // for classes instantiated using shared pointer to refer to itself, equivalent as "this->"
  std::shared_ptr<Vehicle> get_shared_this() { return shared_from_this(); }}


// multi threading
process states: ready, ready suspended (swapped out of main memory towards external storage), blocked (wait for resource to become available), blocked suspended; running
threads: new, runnable (running or rdy to run), blocked (waiting for I/O operations to complete), terminated(finished)
thread scheduler assigns CPU time to Runnable, reactivates Blocked;

std::this_thread::get_id() // main function's thread
unsigned int nThreads = std::thread::hardware_concurrency(); // number of threads supported

void threadFunction(){std::this_thread::sleep_for(std::chrono::milliseconds(100));}
int main(){ std::thread t(threadFunction);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    t.join(); // wait for the thread t to be finished, then proceed further with main, also serves as destructor
    return 0;}
need to compile with: g++ example_3.cpp -pthread
t.detach()// ask main not to wait for threadfunction to finish, cannot be joined ever again

the most vexing parse: anything that couldbe considered as a function declaration, the compiler should parse it as a function declaration, even if it could be interpreted as something else

class Vehicle
{ void operator()(){std::cout << "Vehicle object has been created \n" << std::endl;}};
std::thread t(Vehicle()); // result in error, non-class type, compiler interpreted it as pointer to function returning an object of type Vehicle
use the following instead to initiate an instance of class Vehicle and cout the message above
std::thread t1( (Vehicle()) ); // Add an extra pair of parantheses
std::thread t2 = std::thread( Vehicle() ); // Use copy initialization
std::thread t3{ Vehicle() };// Use uniform initialization with braces

if also the constructor exists: Vehicle(int id) : _id(id) {}
std::thread t = std::thread(Vehicle(1)); would print the cout message above with its ID if adding the ID into the message

// lambda functions
auto f1 = [id]() { std::cout << "ID = " << id << std::endl; };
auto f2 = [&id]() { std::cout << "ID = " << id << std::endl; };
auto f4 = [id]() mutable { std::cout << "ID = " << ++id << std::endl; }; // if u want to change the var in capture list
[]...capture list, capturing external defined vars, ()... parameter list like regular functions

auto f3 = [](const int id) { std::cout << "f) ID in Lambda = " << id << std::endl; };
f3(++id);
id will increase by 1 even outside the scope due to "++id"

auto f0 = [&id]() {};
std::thread t1(f0); if passed by reference, if id is changed before it is processed in the thread, this thread will reflect the change of the id
std::thread t2([id]() mutable {}); // easiest to use, std::thread t2([id] mutable {}) would work too
std::for_each(_threads.begin(), _threads.end(), [](std::thread &t) {t.join();});

// variadic and class methods
when passing to a thread, left value usually are copied and right values are usually moved
in variadic template, left values can also be moved by using std::move()
std::thread t2(printName, std::move(name2), 100); // str and int are parameters of printName, name2 is empty after getting moved
void printName(std::string &name, int waitTime){}
std::thread t(printName, std::ref(name), 50); // need to explicitly use std::ref to make it reference

class Vehicle {void addID(int id) { _id = id; }}
Vehicle v1, v2;
std::thread t1 = std::thread(&Vehicle::addID, v1, 1); // doesn't affect original ID
std::thread t2 = std::thread(&Vehicle::addID, &v2, 2); // affect ID
std::shared_ptr<Vehicle> v(new Vehicle);
std::thread t = std::thread(&Vehicle::addID, v, 1); // affect ID

// vector of threads
std::vector<std::thread> threads;
// copy wouldn't work:
// std::thread t(printHello);
// threads.push_back(t); as copy constructor is forbidden
threads.emplace_back(std::thread(printHello)); // emplace back internally use move semantics
_threads.emplace_back(std::thread(&Vehicle::drive, this)); // implemented in vehicle's own member functions
for (auto &t : threads)t.join();

// future and promise
void modifyMessage(std::promise<std::string> && prms, std::string message){prms.set_value(modifiedMessage);}
int main(){std::promise<std::string> prms;
           std::future<std::string> ftr = prms.get_future();
           std::thread t(modifyMessage, std::move(prms), messageToThread);
           std::string messageFromThread = ftr.get();} // block until get is called
// after prms.set_value(), ftr.get() can then be called
auto status = ftr.wait_for(std::chrono::milliseconds(1000));
status == std::future_status::ready
status == std::future_status::timeout // expired
status == std::future_status::deferred // has been deferred
// passing exception
void divideByNumber(xxx){try{if (denom == 0) throw std::runtime_error("Exception from thread: Division by zero!");
    else prms.set_value(num / denom);}
catch (...){prms.set_exception(std::current_exception());}}// current exception catches the exception right now
// main
try{double result = ftr.get();
    std::cout << "Result = " << result << std::endl;}
catch (std::runtime_error e){std::cout << e.what() << std::endl;}

// using future and async -> tasks (less boiler plate code and hide away a lot of implementation details, )
double divideByNumber(double num, double denom){if (denom == 0)
    throw std::runtime_error("Exception from thread: Division by zero!");
    return num / denom;}
std::future<double> ftr = std::async(divideByNumber, num, denom);// then use the try block above, no need to use t.join() for destruction, system decides whether to execute asynchrounous
std::future<double> ftr = std::async(std::launch::deferred, divideByNumber, num, denom); // synchronous, i.e. same thread ID as main; std::launch::async=>multithread; "any" is default, up to system
std::async([](Vehicle v) {v.setID(2)}, v0); //lambda function and add parameter for the lambda function
thread: handling latency (avoid programs to be blocked); tasks: throughput, exexcuted in parallel

std::vector<std::future<void>> futures;
for (int i = 0; i < nThreads; ++i){futures.emplace_back(std::async(workerFunction, nLoops));}
for (const std::future<void> &ftr : futures) ftr.wait();

// data race: can be prevented by make a copy of original argument using deep copy constructor and passing into lambda function
Vehicle(Vehicle const &src){_name = new std::string;
    *_name = *src._name;}
// move constructor
Vehicle(Vehicle && src){
    _id = src.getID();
    _name = new std::string(src.getName());
    src.setID(0);
    src.setName("Default Name")};
std::future<void> ftr = std::async([](Vehicle v) {v.setName("Vehicle 2")},std::move(v0));
Vehicle(Vehicle && src) : _name(std::move(src._name)){//alternatively using std::move if _name is a unique pointer, origin will be destoyed
    _id = src.getID();
    src.setID(0);};

std::future<void> ftr = std::async(&Intersection::addVehicleToQueue, _currDestination, get_shared_this()); // reference on the function, as _currDestination is pointer to an Intersection; get_shared_this() => "this" for shared ptr

// mutex built in class
std::mutex _mutex; // private member
void pushBack(Vehicle &&v){//member function
_mutex.lock();
_vehicles.emplace_back(std::move(v)); // data race would cause an exception
_mutex.unlock();}
// main
std::shared_ptr<WaitingVehicles> queue(new WaitingVehicles);
std::vector<std::future<void>> futures;
for (int i = 0; i < 1000; ++i){Vehicle v(i);
    futures.emplace_back(std::async(std::launch::async, &WaitingVehicles::pushBack, queue, std::move(v)));}
std::for_each(futures.begin(), futures.end(), [](std::future<void> &ftr) {ftr.wait();});

std::timed_mutex _mutex;
for (int i = 0; i < 3; i++){
    if (_mutex.try_lock_for(std::chrono::milliseconds(100))){ // the number here doesn't matter if the mutex got released before the number of seconds here
        // std::this_thread::sleep_for(std::chrono::milliseconds(100)); this line will cause data race
        _vehicles.emplace_back(std::move(v));
        _mutex.unlock();
        break;} else {
        std::cout <<"failed " << v.getid() << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));}}

// deadlock happens when one thread throws exception and cannot be unlocked or two threads try to use mutex locks which are alrdy used by the other
std::cout can also be resource which can be protected by mutex

std::mutex mtx;
std::lock_guard<std::mutex> lck(mtx);// replacing _mutex.lock(), _mutex.unlock(), gets destoyed as soon as this object is out of scope, exception won't cause deadlock
never use .lock() and .unlock in practice, cos exception will cause deadlock

std::mutex mtx; // cannot be constant
std::unique_lock<std::mutex> lck(mtx); //mostly used, is even better with being able to temporarily unlock and lock again
lck.unlock(); // necessary if stopping processing the shared object
lck.lock();

// handling situation where deadlocks happen when two threads try to use mutex locks which are alrdy used by the other
std::mutex mutex1, mutex2;
std::lock(mutex1, mutex2); // making sure both locks are locked at the same time
std::lock_guard<std::mutex> lock2(mutex2, std::adopt_lock); // both are locked, std::adopt_lock enables std::lock_guard to be applied on alrdy locked mutex

// continuous loop checking whether the queue has received an item
// WaitingVehicles class
bool dataIsAvailable(){std::lock_guard<std::mutex> myLock(_mutex);
  return !_vehicles.empty();}
Vehicle popBack(){std::lock_guard<std::mutex> uLock(_mutex);
  Vehicle v = std::move(_vehicles.back());
  _vehicles.pop_back();
  return v;} // will not be copied due to return value optimization (RVO) in C++
// in main
while (true){// can use "break" to break out an infinite loop
  if (queue->dataIsAvailable())
  {Vehicle v = queue->popBack();
   std::cout << "   Vehicle #" << v.getID() << " has been removed from the queue" << std::endl;}}

// conditional variables
private:std::mutex _mutex;
        std::condition_variable _cond;
Vehicle popBack(){std::unique_lock<std::mutex> uLock(_mutex);
        _cond.wait(uLock, [this] { return !_vehicles.empty(); }); // lambda function, checking _vehicles while locked, then free th access and enter wait if condition not met (spurious wakeup doesn't affect as condition is checked)
                                                                  // if condition is met, lock and proceed with the statements below
        Vehicle v = std::move(_vehicles.back());
        XXX}
void pushBack(Vehicle &&v){std::lock_guard<std::mutex> uLock(_mutex);
        _vehicles.push_back(std::move(v));
        _cond.notify_one();} // notify client after pushing new Vehicle into vector
while (true){Vehicle v = queue->popBack();} // wait state is less burdening for processor
















































// header file, so that in cpp file function definition orders will not matter
#ifndef VECT_ADD_ONE_H
#define VECT_ADD_ONE_H
#include <vector>
using std::vector;
void AddOneToEach(vector<int> &v);
#endif
// cpp
#include "vect_add_one.h"
void AddOneToEach(vector<int> &v)
// 2nd header
#ifndef INCREMENT_AND_SUM_H
#define INCREMENT_AND_SUM_H
int IncrementAndComputeVectorSum(vector<int> v);
#endif
// 2nd cpp
#include "vect_add_one.h"
int IncrementAndComputeVectorSum(vector<int> v) {
    AddOneToEach(v);
}
// main cpp
#include <iostream>
#include <vector>
#include "increment_and_sum.h" // only header file is needed
using std::vector;
using std::cout;
int main()
{
    int total = IncrementAndComputeVectorSum(v);
}

// class header
#ifndef CAR_H
#define CAR_H
#include <string>
using std::string;
using std::cout;
class Car {
  public:
    void PrintCarData();
    Car(string c, int n) : color(c), number(n){}
  private:
    string color;
    int number;
};
#endif
// class cpp
#include <iostream>
#include "car.h"
void Car::PrintCarData(){}
void Car::IncrementDistance(){}
//main cpp
#include <iostream>
#include <string>
#include "car.h"
using std::string;
using std::cout;
int main()
{
    Car car_1 = Car("green", 1);
    car_1.IncrementDistance();
    car_1.PrintCarData();

}
// compile: g++ -std=c++17 ./code/main.cpp ./code/increment_and_sum.cpp ./code/vect_add_one.cpp
// or g++ -c *.cpp, g++ *.o, ./a.out, if one file changes, only need to g++ -c this one file, and then g++ *.o
// cmake on windows: install visual studio build tools, see vscode-cmake-hello subfolder, need to click "search for Kit" if no active kit, then click "CMake: Debug Ready", and then the "bug" symbol
// control shift p: task configure, and then cmake build (make sure cmake block is added to tasks.json)

// cmake on linux: create a folder containing CMakeLists.txt and src/various header and source codes, create a build folder, within which execute cmake .. and make (later on only "make" is needed)
// CMakeLists.txt:
// cmake_minimum_required(VERSION 3.5.1)
// set(CMAKE_CXX_STANDARD 14)
// project(p1)
// add_executable(p1  src/main.cpp  src/vect_add_one.cpp src/increment_and_sum.cpp)
