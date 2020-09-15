using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNN
{
    [Serializable]
    class Point
    {
        public int x;
        public int y;
        public int state;

        public Point(int x, int y, int state) : this(x, y)
        {
            this.state = state;
        }
        public Point(int x, int y)
        {
            this.x = x;
            this.y = y;
            state = 0;
        }
        public Point()
        {
            x = 0;
            y = 0;
            state = 0;
        }

        public Point(Point re)
        {
            x = re.x;
            y = re.y;
            state = re.state;
        }

        public static Point operator +(Point a, Point b)
        {
            return new Point(a.x + b.x, a.y + b.y);
        }

        public static bool operator ==(Point a, Point b)
        {
            return (a.x == b.x && a.y == b.y);
        }
        public static bool operator !=(Point a, Point b)
        {
            return !(a.x != b.x || a.y != b.y);
        }

        public override bool Equals(object obj)
        {
            return base.Equals(obj);
        }
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
        public override string ToString()
        {
            return $"x = {x}  y = {y} ";
        }
    }
}
